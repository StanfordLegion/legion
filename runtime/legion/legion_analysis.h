/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_ANALYSIS_H__
#define __LEGION_ANALYSIS_H__

#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct ContextCoordinate
     * A struct that can uniquely identify an operation inside
     * the context of a parent task by the context_index which
     * is the number of the operation in the context, and the 
     * index_point specifying which point in the case of an
     * index space operation
     */
    struct ContextCoordinate {
      inline ContextCoordinate(void) : context_index(SIZE_MAX) { }
      // Prevent trivally copying for serialize/deserialize
      inline ContextCoordinate(const ContextCoordinate &rhs)
        : context_index(rhs.context_index), index_point(rhs.index_point) { }
      inline ContextCoordinate(ContextCoordinate &&rhs)
        : context_index(rhs.context_index), index_point(rhs.index_point) { }
      inline ContextCoordinate(size_t index, const DomainPoint &p)
        : context_index(index), index_point(p) { }
      inline ContextCoordinate& operator=(const ContextCoordinate &rhs)
        { context_index = rhs.context_index; 
          index_point = rhs.index_point; return *this; }
      inline ContextCoordinate& operator=(ContextCoordinate &&rhs)
        { context_index = rhs.context_index; 
          index_point = rhs.index_point; return *this; }
      inline bool operator==(const ContextCoordinate &rhs) const
        { return ((context_index == rhs.context_index) && 
                  (index_point == rhs.index_point)); }
      inline bool operator<(const ContextCoordinate &rhs) const
        { if (context_index < rhs.context_index) return true;
          if (context_index > rhs.context_index) return false;
          return index_point < rhs.index_point; }
      inline void serialize(Serializer &rez) const
        { rez.serialize(context_index); rez.serialize(index_point); }
      inline void deserialize(Deserializer &derez)
        { derez.deserialize(context_index); derez.deserialize(index_point); }
      size_t context_index;
      DomainPoint index_point;
    };

    /**
     * \struct LogicalUser
     * A class for representing logical users of a logical 
     * region including the necessary information to
     * register mapping dependences on the user.
     */
    struct LogicalUser : public Collectable {
    public:
      LogicalUser(Operation *o, unsigned id, const RegionUsage &u,
                  ProjectionSummary *proj = NULL);
      LogicalUser(const LogicalUser &rhs) = delete;
      ~LogicalUser(void);
    public:
      LogicalUser& operator=(const LogicalUser &rhs) = delete;
    public:
      // For providing deterministic ordering of users in sorted
      // sets which is crucial for control replication
      inline bool deterministic_pointer_less(const LogicalUser *rhs) const
        {
          if (ctx_index < rhs->ctx_index)
            return true;
          if (ctx_index > rhs->ctx_index)
            return false;
          return (idx < rhs->idx);
        }
      inline bool has_timed_out(void)
        {
#ifdef DEBUG_LEGION
          assert(timeout < TIMEOUT);
#endif
          if (++timeout == TIMEOUT)
          {
            timeout = 0;
            return true;
          }
          else
            return false;
        }
    public:
      const RegionUsage usage;
      Operation *const op;
      const size_t ctx_index;
      const unsigned idx;
      const GenerationID gen;
      ProjectionSummary *const shard_proj;
      // This field addresses a problem regarding when
      // to prune tasks out of logical region tree data
      // structures.  If no later task ever performs a
      // dependence test against this user, we might
      // never prune it from the list.  This timeout
      // prevents that from happening by forcing a
      // test to be performed whenever the timeout
      // reaches zero.
      unsigned timeout;
#ifdef LEGION_SPY
      UniqueID uid;
#endif
    public:
      static constexpr unsigned TIMEOUT = LEGION_DEFAULT_LOGICAL_USER_TIMEOUT;
      static_assert(TIMEOUT > 0,
          "LEGION_DEFAULT_LOGICAL_USER_TIMEOUT must be positive");
    };

    /**
     * \class VersionInfo
     * A class for tracking version information about region usage
     */
    class VersionInfo : public LegionHeapify<VersionInfo> {
    public:
      VersionInfo(void);
      VersionInfo(const VersionInfo &rhs);
      virtual ~VersionInfo(void);
    public:
      VersionInfo& operator=(const VersionInfo &rhs);
    public:
      inline bool has_version_info(void) const 
        { return !equivalence_sets.empty(); }
      inline const FieldMaskSet<EquivalenceSet>& get_equivalence_sets(void)
        const { return equivalence_sets; }
      inline void swap(FieldMaskSet<EquivalenceSet> &others)
        { equivalence_sets.swap(others); }
      inline const FieldMask& get_valid_mask(void) const
        { return equivalence_sets.get_valid_mask(); }
      inline void relax_valid_mask(const FieldMask &mask)
        { equivalence_sets.relax_valid_mask(mask); }
    public:
      void pack_equivalence_sets(Serializer &rez) const;
      void unpack_equivalence_sets(Deserializer &derez, Runtime *runtime,
                                   std::set<RtEvent> &ready_events);
    public:
      void record_equivalence_set(EquivalenceSet *set, const FieldMask &mask);
      void clear(void);
    protected:
      FieldMaskSet<EquivalenceSet> equivalence_sets;
    };

    /**
     * \struct LogicalTraceInfo
     * Information about tracing needed for logical
     * dependence analysis.
     */
    struct LogicalTraceInfo {
    public:
      LogicalTraceInfo(Operation *op, unsigned idx,
                       const RegionRequirement &r);
    public:
      LogicalTrace *const trace;
      const unsigned req_idx;
      const RegionRequirement &req;
      const bool skip_analysis;
    };

    /**
     * \struct UniqueInst
     * A small helper class for uniquely naming a physical
     * instance for the purposes of physical trace recording
     */
    struct UniqueInst {
    public:
      UniqueInst(void);
      UniqueInst(IndividualView *v);
    public:
      inline bool operator<(const UniqueInst &rhs) const
      {
        return (inst_did < rhs.inst_did);
      }
      inline bool operator==(const UniqueInst &rhs) const
      {
        return (inst_did == rhs.inst_did);
      }
      inline bool operator!=(const UniqueInst &rhs) const
        { return !this->operator==(rhs); }
    public:
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
      AddressSpaceID get_analysis_space(void) const;
    public:
      // Distributed ID for the physical manager
      DistributedID inst_did;
      // Distributed ID for the view to the instance
      DistributedID view_did;
      // Logical owner space for the view
      AddressSpaceID analysis_space;
#ifdef LEGION_SPY
      RegionTreeID tid;
#endif
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
      virtual void pack_recorder(Serializer &rez, 
                                 std::set<RtEvent> &applied) = 0; 
    public:
      virtual void record_completion_event(ApEvent lhs,
                             unsigned op_kind, const TraceLocalID &tlid) = 0;
      virtual void record_replay_mapping(ApEvent lhs, unsigned op_kind,
                           const TraceLocalID &tlid, bool register_memo) = 0;
      virtual void request_term_event(ApUserEvent &term_event) = 0;
      virtual void record_create_ap_user_event(ApUserEvent &lhs, 
                                               const TraceLocalID &tlid) = 0;
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        const TraceLocalID &tlid) = 0;
    public:
      virtual void record_merge_events(ApEvent &lhs, ApEvent rhs,
                                       const TraceLocalID &tlid) = 0;
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       const TraceLocalID &tlid) = 0;
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       ApEvent e3,const TraceLocalID &tlid) = 0;
      virtual void record_merge_events(ApEvent &lhs,
                                       const std::set<ApEvent>& rhs,
                                       const TraceLocalID &tlid) = 0;
      virtual void record_merge_events(ApEvent &lhs,
                                       const std::vector<ApEvent>& rhs,
                                       const TraceLocalID &tlid) = 0;
      virtual void record_merge_events(PredEvent &lhs,
                                       PredEvent e1, PredEvent e2,
                                       const TraceLocalID &tlid) = 0;
      // This collective barrier is managed by the operations and is auto
      // refreshed by the operations on each replay
      virtual void record_collective_barrier(ApBarrier bar, ApEvent pre,
                 const std::pair<size_t,size_t> &key, size_t arrival_count) = 0;
      // This collective barrier is managed by the template and will be
      // refreshed as necessary when barrier generations are exhausted
      virtual ShardID record_managed_barrier(ApBarrier bar,
                                             size_t total_arrivals) = 0;
      virtual void record_barrier_arrival(ApBarrier bar, ApEvent pre,
                    size_t arrival_count, std::set<RtEvent> &applied,
                    ShardID owner_shard) = 0;
    public:
      virtual void record_issue_copy(const TraceLocalID &tlid, 
                           ApEvent &lhs, IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
                           const std::vector<Reservation>& reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent src_unique, LgEvent dst_unique,
                           int priority, CollectiveKind collective) = 0;
      virtual void record_issue_across(const TraceLocalID &tlid, ApEvent &lhs,
                           ApEvent collective_precondition, 
                           ApEvent copy_precondition,
                           ApEvent src_indirect_precondition,
                           ApEvent dst_indirect_precondition,
                           CopyAcrossExecutor *executor) = 0;
      virtual void record_copy_insts(ApEvent lhs, const TraceLocalID &tlid,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const UniqueInst &src_inst, 
                           const UniqueInst &dst_inst,
                           const FieldMask &src_mask, const FieldMask &dst_mask,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           ReductionOpID redop, std::set<RtEvent> &applied) = 0;
      typedef LegionMap<UniqueInst,FieldMask> AcrossInsts;
      virtual void record_across_insts(ApEvent lhs, const TraceLocalID &tlid,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const AcrossInsts &src_insts,
                           const AcrossInsts &dst_insts,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           bool src_indirect, bool dst_indirect,
                           std::set<RtEvent> &applied) = 0;
      virtual void record_indirect_insts(ApEvent indirect_done,
                           ApEvent all_done, IndexSpaceExpression *expr,
                           const AcrossInsts &insts, 
                           std::set<RtEvent> &applied, PrivilegeMode priv) = 0;
      virtual void record_issue_fill(const TraceLocalID &tlid, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent unique_event, int priority, 
                           CollectiveKind collective) = 0;
      virtual void record_fill_inst(ApEvent lhs, IndexSpaceExpression *expr,
                           const UniqueInst &dst_inst,
                           const FieldMask &fill_mask,
                           std::set<RtEvent> &applied_events,
                           const bool reduction_initialization) = 0;
    public:
      virtual void record_op_inst(const TraceLocalID &tlid,
                          unsigned parent_req_index,
                          const UniqueInst &inst,
                          RegionNode *node,
                          const RegionUsage &usage,
                          const FieldMask &user_mask,
                          bool update_validity,
                          std::set<RtEvent> &applied) = 0;
      virtual void record_set_op_sync_event(ApEvent &lhs,
                          const TraceLocalID &tlid) = 0;
      virtual void record_mapper_output(const TraceLocalID &tlid,
                         const Mapper::MapTaskOutput &output,
                         const std::deque<InstanceSet> &physical_instances,
                         const std::vector<size_t> &future_size_bounds,
                         const std::vector<TaskTreeCoordinates> &coordinates,
                         std::set<RtEvent> &applied_events) = 0;
      virtual void record_complete_replay(const TraceLocalID &tlid,
                                          ApEvent pre, ApEvent post,
                                          std::set<RtEvent> &applied) = 0;
      virtual void record_reservations(const TraceLocalID &tlid,
                                const std::map<Reservation,bool> &locks,
                                std::set<RtEvent> &applied_events) = 0;
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
        REMOTE_TRACE_RECORD_COMPLETION_EVENT,
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
      RemoteTraceRecorder(Runtime *rt, AddressSpaceID origin,AddressSpace local,
                          const TraceLocalID &tlid, PhysicalTemplate *tpl, 
                          RtUserEvent applied_event);
      RemoteTraceRecorder(const RemoteTraceRecorder &rhs) = delete;
      virtual ~RemoteTraceRecorder(void);
    public:
      RemoteTraceRecorder& operator=(const RemoteTraceRecorder &rhs) = delete;
    public:
      virtual bool is_recording(void) const { return true; }
      virtual void add_recorder_reference(void);
      virtual bool remove_recorder_reference(void);
      virtual void pack_recorder(Serializer &rez, 
                                 std::set<RtEvent> &applied);
    public:
      virtual void record_completion_event(ApEvent lhs, unsigned op_kind,
                                           const TraceLocalID &tlid);
      virtual void record_replay_mapping(ApEvent lhs, unsigned op_kind,
                           const TraceLocalID &tlid, bool register_memo);
      virtual void request_term_event(ApUserEvent &term_event);
      virtual void record_create_ap_user_event(ApUserEvent &hs, 
                                               const TraceLocalID &tlid);
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        const TraceLocalID &tlid);
    public:
      virtual void record_merge_events(ApEvent &lhs, ApEvent rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       ApEvent e3, const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::set<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::vector<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(PredEvent &lhs,
                                       PredEvent e1, PredEvent e2,
                                       const TraceLocalID &tlid);
      virtual void record_collective_barrier(ApBarrier bar, ApEvent pre,
                    const std::pair<size_t,size_t> &key, size_t arrival_count);
      virtual ShardID record_managed_barrier(ApBarrier bar,
                                             size_t total_arrivals);
      virtual void record_barrier_arrival(ApBarrier bar, ApEvent pre,
                    size_t arrival_count, std::set<RtEvent> &applied,
                    ShardID owner_shard);
    public:
      virtual void record_issue_copy(const TraceLocalID &tlid, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent src_unique, LgEvent dst_unique,
                           int priority, CollectiveKind collective);
      virtual void record_issue_across(const TraceLocalID &tlid, ApEvent &lhs,
                           ApEvent collective_precondition, 
                           ApEvent copy_precondition,
                           ApEvent src_indirect_precondition,
                           ApEvent dst_indirect_precondition,
                           CopyAcrossExecutor *executor);
      virtual void record_copy_insts(ApEvent lhs, const TraceLocalID &tlid,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const UniqueInst &src_inst,
                           const UniqueInst &dst_inst,
                           const FieldMask &src_mask, const FieldMask &dst_mask,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           ReductionOpID redop, std::set<RtEvent> &applied);
      virtual void record_across_insts(ApEvent lhs, const TraceLocalID &tlid,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const AcrossInsts &src_insts,
                           const AcrossInsts &dst_insts,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           bool src_indirect, bool dst_indirect,
                           std::set<RtEvent> &applied);
      virtual void record_indirect_insts(ApEvent indirect_done,ApEvent all_done,
                           IndexSpaceExpression *expr,
                           const AcrossInsts &insts,
                           std::set<RtEvent> &applied, PrivilegeMode priv);
      virtual void record_issue_fill(const TraceLocalID &tlid, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent unique_event, int priority,
                           CollectiveKind collective);
      virtual void record_fill_inst(ApEvent lhs, IndexSpaceExpression *expr,
                           const UniqueInst &dst_inst,
                           const FieldMask &fill_mask,
                           std::set<RtEvent> &applied_events,
                           const bool reduction_initialization);
    public:
      virtual void record_op_inst(const TraceLocalID &tlid,
                          unsigned parent_req_index,
                          const UniqueInst &inst,
                          RegionNode *node,
                          const RegionUsage &usage,
                          const FieldMask &user_mask,
                          bool update_validity,
                          std::set<RtEvent> &applied);
      virtual void record_set_op_sync_event(ApEvent &lhs,
                          const TraceLocalID &tlid);
      virtual void record_mapper_output(const TraceLocalID &tlid,
                          const Mapper::MapTaskOutput &output,
                          const std::deque<InstanceSet> &physical_instances,
                          const std::vector<size_t> &future_size_bounds,
                          const std::vector<TaskTreeCoordinates> &coordinates,
                          std::set<RtEvent> &applied_events);
      virtual void record_complete_replay(const TraceLocalID &tlid,
                                          ApEvent pre, ApEvent post,
                                          std::set<RtEvent> &applied);
      virtual void record_reservations(const TraceLocalID &tlid,
                                const std::map<Reservation,bool> &locks,
                                std::set<RtEvent> &applied_events);
    public:
      static RemoteTraceRecorder* unpack_remote_recorder(Deserializer &derez,
                                    Runtime *runtime, const TraceLocalID &tlid);
      static void handle_remote_update(Deserializer &derez, 
                  Runtime *runtime, AddressSpaceID source);
      static void handle_remote_response(Deserializer &derez);
    protected:
      static void pack_src_dst_field(Serializer &rez, const CopySrcDstField &f);
      static void unpack_src_dst_field(Deserializer &derez, CopySrcDstField &f);
    public:
      Runtime *const runtime;
      const AddressSpaceID origin_space;
      const AddressSpaceID local_space;
      PhysicalTemplate *const remote_tpl;
      const RtUserEvent applied_event;
    protected:
      mutable LocalLock applied_lock;
      std::set<RtEvent> applied_events;
    };

    /**
     * \struct TraceInfo
     * This provides a generic tracing struct for operations
     */
    struct TraceInfo {
    public:
      explicit TraceInfo(Operation *op);
      TraceInfo(SingleTask *task, RemoteTraceRecorder *rec); 
      TraceInfo(const TraceInfo &info);
      ~TraceInfo(void);
    protected:
      TraceInfo(PhysicalTraceRecorder *rec,
                const TraceLocalID &tlid);
    public:
      inline void record_replay_mapping(ApEvent lhs, unsigned op_kind,
                                        bool register_memo) const
        {
          base_sanity_check();
          rec->record_replay_mapping(lhs, op_kind, tlid, register_memo);
        }
      inline void request_term_event(ApUserEvent &term_event) const
        {
          base_sanity_check();
          rec->request_term_event(term_event);
        }
      inline void record_create_ap_user_event(ApUserEvent &result) const
        {
          base_sanity_check();
          rec->record_create_ap_user_event(result, tlid);
        }
      inline void record_trigger_event(ApUserEvent result, ApEvent rhs) const
        {
          base_sanity_check();
          rec->record_trigger_event(result, rhs, tlid);
        }
      inline void record_merge_events(PredEvent &result,
                                      PredEvent e1, PredEvent e2) const
        {
          base_sanity_check();
          rec->record_merge_events(result, e1, e2, tlid);
        }
      inline void record_merge_events(ApEvent &result, 
                                      ApEvent e1, ApEvent e2) const
        {
          base_sanity_check();
          rec->record_merge_events(result, e1, e2, tlid);
        }
      inline void record_merge_events(ApEvent &result, ApEvent e1, 
                                      ApEvent e2, ApEvent e3) const
        {
          base_sanity_check();
          rec->record_merge_events(result, e1, e2, e3, tlid);
        }
      inline void record_merge_events(ApEvent &result, 
                                      const std::set<ApEvent> &events) const
        {
          base_sanity_check();
          rec->record_merge_events(result, events, tlid);
        }
      inline void record_merge_events(ApEvent &result, 
                                      const std::vector<ApEvent> &events) const
        {
          base_sanity_check();
          rec->record_merge_events(result, events, tlid);
        }
      inline void record_collective_barrier(ApBarrier bar, ApEvent pre,
           const std::pair<size_t,size_t> &key, size_t arrival_count = 1) const
        {
          base_sanity_check();
          rec->record_collective_barrier(bar, pre, key, arrival_count);
        }
      inline ShardID record_managed_barrier(ApBarrier bar, 
                                            size_t total_arrivals) const
        {
          base_sanity_check();
          return rec->record_managed_barrier(bar, total_arrivals);
        }
      inline void record_barrier_arrival(ApBarrier bar, ApEvent pre,
          size_t arrival_count, std::set<RtEvent> &applied, ShardID owner) const
        {
          base_sanity_check();
          rec->record_barrier_arrival(bar, pre, arrival_count, applied, owner);
        }
      inline void record_op_sync_event(ApEvent &result) const
        {
          base_sanity_check();
          rec->record_set_op_sync_event(result, tlid);
        }
      inline void record_mapper_output(const TraceLocalID &tlid, 
                          const Mapper::MapTaskOutput &output,
                          const std::deque<InstanceSet> &physical_instances,
                          const std::vector<size_t> &future_size_bounds,
                          const std::vector<TaskTreeCoordinates> &coordinates,
                          std::set<RtEvent> &applied)
        {
          base_sanity_check();
          rec->record_mapper_output(tlid, output, physical_instances,
                            future_size_bounds, coordinates, applied);
        }
      inline void record_complete_replay(ApEvent pre, ApEvent post,
                                         std::set<RtEvent> &applied) const
        {
          base_sanity_check();
          rec->record_complete_replay(tlid, pre, post, applied);
        }
      inline void record_reservations(const TraceLocalID &tlid,
                      const std::map<Reservation,bool> &reservations,
                      std::set<RtEvent> &applied) const
        {
          base_sanity_check();
          rec->record_reservations(tlid, reservations, applied);
        }
    protected:
      inline void base_sanity_check(void) const
        {
#ifdef DEBUG_LEGION
          assert(recording);
          assert(rec != NULL);
          assert(rec->is_recording());
#endif
        }
      static PhysicalTraceRecorder* init_recorder(Operation *op);
      static TraceLocalID init_tlid(Operation *op);
    protected:
      PhysicalTraceRecorder *const rec;
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
      PhysicalTraceInfo(Operation *op, unsigned index);
      PhysicalTraceInfo(const TraceInfo &info, unsigned index,
                        bool update_validity = true);
      // Weird argument order to help the compiler avoid ambiguity
      PhysicalTraceInfo(unsigned src_idx, const TraceInfo &info, 
                        unsigned dst_idx);
      PhysicalTraceInfo(const PhysicalTraceInfo &rhs);
    protected:
      PhysicalTraceInfo(const TraceLocalID &tlid,
                        unsigned src_idx, unsigned dst_idx,
                        bool update_validity, PhysicalTraceRecorder *rec);
    public:
      inline void record_issue_copy(ApEvent &result,
                          IndexSpaceExpression *expr,
                          const std::vector<CopySrcDstField>& src_fields,
                          const std::vector<CopySrcDstField>& dst_fields,
                          const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                          ApEvent precondition, PredEvent pred_guard,
                          LgEvent src_unique, LgEvent dst_unique,
                          int priority, CollectiveKind collective) const
        {
          sanity_check();
          rec->record_issue_copy(tlid, result, expr, src_fields,
                                 dst_fields, reservations,
#ifdef LEGION_SPY
                                 src_tree_id, dst_tree_id,
#endif
                                 precondition, pred_guard,
                                 src_unique, dst_unique,
                                 priority, collective);
        }
      inline void record_issue_fill(ApEvent &result,
                          IndexSpaceExpression *expr,
                          const std::vector<CopySrcDstField> &fields,
                          const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                          UniqueID fill_uid,
                          FieldSpace handle,
                          RegionTreeID tree_id,
#endif
                          ApEvent precondition, PredEvent pred_guard,
                          LgEvent unique_event, int priority,
                          CollectiveKind collective) const
        {
          sanity_check();
          rec->record_issue_fill(tlid, result, expr, fields, 
                                 fill_value, fill_size,
#ifdef LEGION_SPY
                                 fill_uid, handle, tree_id,
#endif
                                 precondition, pred_guard,
                                 unique_event, priority, collective);
        }
      inline void record_issue_across(ApEvent &result,
                                      ApEvent collective_precondition,
                                      ApEvent copy_precondition,
                                      ApEvent src_indirect_precondition,
                                      ApEvent dst_indirect_precondition,
                                      CopyAcrossExecutor *executor) const
        {
          sanity_check();
          rec->record_issue_across(tlid, result, collective_precondition,
                      copy_precondition, src_indirect_precondition,
                      dst_indirect_precondition, executor);
        }
      inline void record_fill_inst(ApEvent lhs,
                                   IndexSpaceExpression *expr,
                                   const UniqueInst &inst,
                                   const FieldMask &fill_mask,
                                   std::set<RtEvent> &applied,
                                   const bool reduction_initialization) const
        {
          sanity_check();
          rec->record_fill_inst(lhs, expr, inst, fill_mask,
                                applied, reduction_initialization);
        }
      inline void record_copy_insts(ApEvent lhs,
                                    IndexSpaceExpression *expr,
                                    const UniqueInst &src_inst,
                                    const UniqueInst &dst_inst,
                                    const FieldMask &src_mask,
                                    const FieldMask &dst_mask,
                                    ReductionOpID redop,
                                    std::set<RtEvent> &applied) const
        {
          sanity_check();
          rec->record_copy_insts(lhs, tlid, index, dst_index, expr,
                                 src_inst, dst_inst, src_mask, dst_mask, 
                                 LEGION_READ_PRIV, (redop > 0) ?
                                  LEGION_REDUCE_PRIV : LEGION_WRITE_PRIV,
                                 redop, applied);
        }
      typedef LegionMap<UniqueInst,FieldMask> AcrossInsts;
      inline void record_across_insts(ApEvent lhs, unsigned idx1, unsigned idx2,
                                      PrivilegeMode mode1, PrivilegeMode mode2,
                                      IndexSpaceExpression *expr,
                                      AcrossInsts &src_insts,
                                      AcrossInsts &dst_insts,
                                      bool src_indirect, bool dst_indirect,
                                      std::set<RtEvent> &applied) const
        {
          sanity_check();
          rec->record_across_insts(lhs, tlid, idx1, idx2, expr,
                                   src_insts, dst_insts, mode1, mode2,
                                   src_indirect, dst_indirect, applied);
        }
      inline void record_indirect_insts(ApEvent indirect_done, ApEvent all_done,
                                        IndexSpaceExpression *expr,
                                        AcrossInsts &insts,
                                        std::set<RtEvent> &applied,
                                        PrivilegeMode privilege) const
        {
          sanity_check();
          rec->record_indirect_insts(indirect_done, all_done, expr, insts,
                                     applied, privilege);
        }
      // Not inline because we need to call a method on Operation
      void record_op_inst(const RegionUsage &usage,
                          const FieldMask &user_mask,
                          const UniqueInst &inst,
                          RegionNode *node, Operation *op,
                          std::set<RtEvent> &applied) const;
    public:
      void pack_trace_info(Serializer &rez, std::set<RtEvent> &applied) const;
      static PhysicalTraceInfo unpack_trace_info(Deserializer &derez,
                                                 Runtime *runtime);
    private:
      inline void sanity_check(void) const
        {
#ifdef DEBUG_LEGION
          base_sanity_check();
          assert(index != -1U);
          assert(dst_index != -1U);
#endif
        }
    public:
      const unsigned index;
      const unsigned dst_index;
      const bool update_validity;
    };

    /**
     * \class ProjectionInfo
     * Projection information for index space requirements
     */
    class ProjectionInfo {
    public:
      ProjectionInfo(void)
        : projection(NULL), projection_type(LEGION_SINGULAR_PROJECTION),
          projection_space(NULL) { }
      ProjectionInfo(Runtime *runtime, const RegionRequirement &req,
                     IndexSpaceNode *launch_space,ShardingFunction *func = NULL,
                     IndexSpace shard_space = IndexSpace::NO_SPACE);
    public:
      inline bool is_projecting(void) const { return (projection != NULL); }
      inline bool is_sharding(void) const { return (sharding_function != NULL); }
      bool is_complete_projection(RegionTreeNode *node,
                                  const LogicalUser &user) const;
    public:
      ProjectionFunction *projection;
      ProjectionType projection_type;
      IndexSpaceNode *projection_space;
      ShardingFunction *sharding_function;
      IndexSpaceNode *sharding_space;
    };

    /**
     * \struct PhysicalUser
     * A class for representing physical users of a logical
     * region including necessary information to 
     * register execution dependences on the user.
     */
    struct PhysicalUser : public Collectable {
    public:
      static const AllocationType alloc_type = PHYSICAL_USER_ALLOC;
    public:
      PhysicalUser(const RegionUsage &u, IndexSpaceExpression *expr,
                   UniqueID op_id, unsigned index, bool copy, bool covers);
      PhysicalUser(const PhysicalUser &rhs);
      ~PhysicalUser(void);
    public:
      PhysicalUser& operator=(const PhysicalUser &rhs);
    public:
      void pack_user(Serializer &rez, const AddressSpaceID target) const;
      static PhysicalUser* unpack_user(Deserializer &derez, 
              RegionTreeForest *forest, const AddressSpaceID source);
    public:
      const RegionUsage usage;
      IndexSpaceExpression *const expr;
      const UniqueID op_id;
      const unsigned index; // region requirement index
      const bool copy_user; // is this from a copy or an operation
      const bool covers; // whether the expr covers the ExprView its in
    };  

#if 0
    /**
     * \struct ProjectionSummary
     * A small helper class that tracks the triple that 
     * uniquely defines a set of region requirements
     * for a projection operation
     */
    struct ProjectionSummary {
    public:
      ProjectionSummary(void);
      ProjectionSummary(IndexSpaceNode *is, 
                        ProjectionFunction *p, 
                        ShardingFunction *s,
                        IndexSpaceNode *sd);
      ProjectionSummary(const ProjectionInfo &info);
      ProjectionSummary(ProjectionSummary &&rhs);
      ProjectionSummary(const ProjectionSummary &rhs);
      ~ProjectionSummary(void);
    public:
      ProjectionSummary& operator=(const ProjectionSummary &rhs);
    public:
      bool operator<(const ProjectionSummary &rhs) const;
      bool operator==(const ProjectionSummary &rhs) const;
      bool operator!=(const ProjectionSummary &rhs) const;
    public:
      void pack_summary(Serializer &rez) const;
      static ProjectionSummary unpack_summary(Deserializer &derez,
                        RegionTreeForest *context);
    public:
      IndexSpaceNode *domain;
      ProjectionFunction *projection;
      ShardingFunction *sharding;
      IndexSpaceNode *sharding_domain;
    };

    /**
     * \struct RefProjectionSummary
     * A refinement projection summary is just a projection summary
     * with support for reference counting and no copies
     */
    struct RefProjectionSummary : public ProjectionSummary, public Collectable {
    public:
      RefProjectionSummary(const ProjectionInfo &info);
      RefProjectionSummary(ProjectionSummary &&rhs);
      RefProjectionSummary(const RefProjectionSummary &rhs);
      ~RefProjectionSummary(void);
    public:
      RefProjectionSummary& operator=(const RefProjectionSummary &rhs);
    public:
      void project_refinement(RegionTreeNode *node, 
                              std::vector<RegionNode*> &regions) const;
      void project_refinement(RegionTreeNode *node, ShardID shard,
                              std::vector<RegionNode*> &regions,
                              Provenance *provenance) const;
    };
#endif

    /**
     * \struct FieldState
     * Track the field state more accurately
     * for logical traversals to figure out 
     * which tasks can run in parallel.
     */
    struct FieldState {
    public:
      FieldState(void);
      FieldState(OpenState state, const FieldMask &m,
                 RegionTreeNode *child);
      FieldState(const RegionUsage &usage, const FieldMask &m, 
                 RegionTreeNode *child);
      FieldState(const FieldState &rhs);
      FieldState(FieldState &&rhs) noexcept;
      FieldState& operator=(const FieldState &rhs);
      FieldState& operator=(FieldState &&rhs) noexcept;
      ~FieldState(void);
    public:
      inline const FieldMask& valid_fields(void) const 
        { return open_children.get_valid_mask(); }
    public:
      bool overlaps(const FieldState &rhs) const;
      void merge(FieldState &rhs, RegionTreeNode *node);
      bool filter(const FieldMask &mask);
      void add_child(RegionTreeNode *child,
                     const FieldMask &mask);
      void remove_child(RegionTreeNode *child);
    public:
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask,
                       RegionNode *node) const;
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask,
                       PartitionNode *node) const;
    public:
      typedef FieldMaskSet<RegionTreeNode,UNTRACKED_ALLOC,true/*ordered*/>
        OrderedFieldMaskChildren;
      OrderedFieldMaskChildren open_children;
      OpenState open_state;
      ReductionOpID redop;
    };

    /**
     * \class ShardedColorMap
     * This data structure is a look-up table for mapping colors in a
     * disjoint and complete partition to the nearest shard that knows
     * about them. It is the only data structure that we create anywhere
     * which might be on the order of the number of nodes/shards and 
     * stored on every node so we deduplicate it across all its uses.
     */
    class ShardedColorMap : public Collectable {
    public:
      ShardedColorMap(std::unordered_map<LegionColor,ShardID> &&map)
        : color_shards(map) { }
    public:
      inline bool empty(void) const { return color_shards.empty(); }
      inline size_t size(void) const { return color_shards.size(); }
      ShardID at(LegionColor color) const;
      void pack(Serializer &rez);
      static void pack_empty(Serializer &rez);
      static ShardedColorMap* unpack(Deserializer &derez);
    public:
      // Must remain constant since many things can refer to this
      const std::unordered_map<LegionColor,ShardID> color_shards;
    };

#if 0
    /**
     * \class RefinementNode
     * This data captures an actualized refinement tree to pass to a
     * refinement operation to update which regions are being used for
     * equivalence sets. It lives from the time that it is created 
     * up to the point that it is invalidated by the version manager.
     */
    class RefinementNode {
    public:
      virtual ~RefinementNode(void) { }
    public:
      virtual RegionRefinementNode* as_region_refinement(void) { return NULL; }
      virtual PartitionRefinementNode* as_partition_refinement(void)
        { return NULL; };
      virtual RegionTreeNode* get_region_tree_node(void) const = 0;
      virtual RefinementNode* clone(void) const = 0;
      virtual bool incorporate(RefinementNode *to_incorporate) = 0;
      virtual void perform_versioning_analysis(ContextID ctx,
          InnerContext *context, const FieldMask &mask, 
          LegionMap<RegionNode*,VersionInfo> &version_infos,
          UniqueID op_id, std::set<RtEvent> &ready_events) const = 0;
      virtual void register_refinement(ContextID ctx, const FieldMask &mask,
          InnerContext *context, size_t op_ctx_index,unsigned refinement_number,
          unsigned parent_req_index, std::set<RtEvent> &applied_events,
          const LegionMap<RegionNode*,VersionInfo> &version_infos) const = 0;
    public:
      FieldMask increment_touches(const FieldMask &mask);
      FieldMask dominates_touches(const FieldMask &mask,
                                  const RefinementNode *rhs) const;
      void filter_touches(const FieldMask &mask);
      void record_refinement_tree(ContextID ctx, const FieldMask &mask) const;
    public:
      inline size_t count_children(void) const { return children.size(); }
      bool is_mostly_complete(void) const;
      bool matches(const RefinementNode *sibling) const;
      bool matches_child(LegionColor color, RefinementNode *child) const;
      void update_child(LegionColor color, RefinementNode *child);
      RefinementNode* clone(void) const;
    public:
      RegionTreeNode *const node;
      static constexpr unsigned REFINEMENT_CHANGE_COUNT = 16;
    protected:
      std::map<LegionColor,RefinementNode*> children;
      std::map<LegionColor,ShardID> shard_children; // only for partitions
      LegionMap<unsigned,FieldMask> touches;
    };

    /**
     * \class RegionRefinementNode
     * A region refinement node stores the meta-data required for 
     * representing a logical region in a refinement tree
     */
    class RegionRefinementNode : public RefinementNode {
    public:
      RegionRefinementNode(RegionNode *owner,
                           PartitionRefinementNode *child = NULL);
      RegionRefinementNode(const RegionRefinementNode &rhs) = delete;
      virtual ~RegionRefinementNode(void);
    public:
      RegionRefinementNode& operator=(const RegionRefinementNode &rhs) = delete;
    public:
      virtual RegionRefinementNode* as_region_refinement(void) { return this; }
      virtual RegionTreeNode* get_region_tree_node(void) const;
      virtual RefinementNode* clone(void) const;
      virtual bool incorporate(RefinementNode *to_incorporate);
      virtual void perform_versioning_analysis(ContextID ctx,
          InnerContext *context, const FieldMask &mask, 
          LegionMap<RegionNode*,VersionInfo> &version_infos,
          UniqueID op_id, std::set<RtEvent> &ready_events) const;
      virtual void register_refinement(ContextID ctx, const FieldMask &mask,
          InnerContext *context, size_t op_ctx_index,unsigned refinement_number,
          unsigned parent_req_index, std::set<RtEvent> &applied_events,
          const LegionMap<RegionNode*,VersionInfo> &version_infos) const;
    public:
      RegionNode *const node;
      PartitionRefinementNode *child;
      // Should only have refining shards at the leaves of the refinement
      // tree and only in control replicated contexts
      std::vector<ShardID> refining_shards; // this vector is sorted
    };

    /**
     * \class RegionRefinementNode
     * A region refinement node stores the meta-data required for 
     * representing a logical region in a refinement tree
     */
    class PartitionRefinementNode : public RefinementNode {
    public:
      PartitionRefinementNode(PartitionNode *owner,ShardedColorMap *map = NULL);
      PartitionRefinementNode(const PartitionRefinementNode &rhs) = delete;
      virtual ~PartitionRefinementNode(void);
    public:
      PartitionRefinementNode& operator=(
          const PartitionRefinementNode &rhs) = delete;
    public:
      virtual PartitionRefinementNode* as_partition_refinement(void)
        { return this; }
      virtual RegionTreeNode* get_region_tree_node(void) const;
      virtual RefinementNode* clone(void) const;
      virtual bool incorporate(RefinementNode *to_incorporate);
      virtual void perform_versioning_analysis(ContextID ctx,
          InnerContext *context, const FieldMask &mask, 
          LegionMap<RegionNode*,VersionInfo> &version_infos,
          UniqueID op_id, std::set<RtEvent> &ready_events) const;
      virtual void register_refinement(ContextID ctx, const FieldMask &mask,
          InnerContext *context, size_t op_ctx_index,unsigned refinement_number,
          unsigned parent_req_index, std::set<RtEvent> &applied_events,
          const LegionMap<RegionNode*,VersionInfo> &version_infos) const;
    public:
      PartitionNode *const node;
      ShardedColorMap *const children_shards;
      std::unordered_map<LegionColor,RegionRefinementNode*> children;
    };
#endif

    /**
     * \class ProjectionNode
     * A projection node represents a summary of the regions and partitions
     * accessed by a projection function from a particular node in the 
     * region tree. In the case of control replication, it specifically
     * stores the accesses performed by the local shard, and stores at
     * least one NULL child for any aliasing children from different
     * shards to facilitate testing for any close operations that might
     * be required between shards. We can also test a projection node 
     * to see if it can be converted to a refinement node (e.g. that
     * is has all disjoint-complete partitions).
     */
    class ProjectionNode : public Collectable {
    public:
      /**
       * This class defines an interval tree on colors that we can
       * that we can use to test if a projection node interferes with
       * children from a remote shard. The tree is defined for the ranges
       * of colors represented by remote shards. These ranges are 
       * exclusive from our local children. The reason to use 
       * ranges here is to compress the representation of the colors
       * from all the remote shards. We need a way to test if children
       * are represented on remote shards, but we don't care what's
       * in their subtrees. The ranges are semi-inclusive [start,end)
       * Note that the ranges by definition cannot overlap with eachother
       * It's important to realized that the only reason that this 
       * compression works is that we linearize colors in N-d color
       * spaces using Morton codes which gives good locality for 
       * nearest neighbors and encourages this compressibility so
       * we can efficiently store the children as ranges to query.
       */
      class IntervalTree {
      public:
        inline void swap(IntervalTree &rhs) { ranges.swap(rhs.ranges); }
        inline bool empty(void) const { return ranges.empty(); }
        void add_child(LegionColor color);
        void remove_child(LegionColor color);
        void add_range(LegionColor start, LegionColor stop);
        bool has_child(LegionColor color) const;
        void serialize(Serializer &rez) const;
        void deserialize(Deserializer &derez);
      public:
        std::map<LegionColor/*start*/,LegionColor/*end*/> ranges;
      };
      // These structures are used for exchanging summary information
      // between different shards with control replication
      struct RegionSummary {
        ProjectionNode::IntervalTree children;
        std::vector<ShardID> users;
      };
      struct PartitionSummary {
        ProjectionNode::IntervalTree children;
        // If we're disjoint and complete we also track the sets
        // of shards that know about each of the children as well
        // so we can record the one nearest for each shard
        std::multimap<LegionColor,ShardID> disjoint_complete_child_shards;
      };
    public:
      virtual ~ProjectionNode(void) { };
      virtual ProjectionRegion* as_region_projection(void) { return NULL; }
      virtual ProjectionPartition *as_partition_projection(void) 
        { return NULL; }
      virtual bool is_disjoint(void) const = 0;
#if 0
      virtual bool is_complete(void) const = 0;
#endif
      virtual bool is_leaves_only(void) const = 0;
      virtual bool is_unique_shards(void) const = 0;
      virtual bool interferes(ProjectionNode *other, ShardID local) const = 0;
      virtual void extract_shard_summaries(bool supports_name_based_analysis,
          ShardID local_shard, std::map<LogicalRegion,RegionSummary> &regions,
          std::map<LogicalPartition,PartitionSummary> &partitions) const = 0;
      virtual void update_shard_summaries(bool supports_name_based_analysis,
          ShardID local_shard, size_t total_shards,
          std::map<LogicalRegion,RegionSummary> &regions,
          std::map<LogicalPartition,PartitionSummary> &partitions) = 0;
#if 0
      virtual RefinementNode* create_refinement(void) const = 0;
#endif
    public:
      IntervalTree shard_children;
    };

    class ProjectionRegion : public ProjectionNode {
    public:
      ProjectionRegion(RegionNode *node);
      ProjectionRegion(const ProjectionRegion &rhs) = delete;
      virtual ~ProjectionRegion(void);
    public:
      ProjectionRegion& operator=(const ProjectionRegion &rhs) = delete;
    public:
      virtual ProjectionRegion* as_region_projection(void) { return this; }
      virtual bool is_disjoint(void) const;
#if 0
      virtual bool is_complete(void) const;
#endif
      virtual bool is_leaves_only(void) const;
      virtual bool is_unique_shards(void) const;
      virtual bool interferes(ProjectionNode *other, ShardID local) const;
      virtual void extract_shard_summaries(bool supports_name_based_analysis,
          ShardID local_shard, std::map<LogicalRegion,RegionSummary> &regions,
          std::map<LogicalPartition,PartitionSummary> &partitions) const;
      virtual void update_shard_summaries(bool supports_name_based_analysis,
          ShardID local_shard, size_t total_shards,
          std::map<LogicalRegion,RegionSummary> &regions,
          std::map<LogicalPartition,PartitionSummary> &partitions);
#if 0
      virtual RefinementNode* create_refinement(void) const;
#endif
      bool has_interference(ProjectionRegion *other, ShardID local) const;
      void add_user(ShardID shard);
      void add_child(ProjectionPartition *child);
#if 0
      void convert(const std::vector<ShardID> *shard_to_shard_mapping);
      void merge(ProjectionRegion *rhs,
                 const std::vector<ShardID> *shard_to_shard_mapping);
      void pack(Serializer &rez) const;
      void unpack(Deserializer &derez);
#endif
    public:
      RegionNode *const region;
      std::unordered_map<LegionColor,ProjectionPartition*> local_children;
      std::vector<ShardID> shard_users; // this vector is sorted
    }; 

    class ProjectionPartition : public ProjectionNode {
    public:
      ProjectionPartition(PartitionNode *node, ShardedColorMap *map = NULL);
      ProjectionPartition(const ProjectionPartition &rhs) = delete;
      virtual ~ProjectionPartition(void);
    public:
      ProjectionPartition& operator=(const ProjectionPartition &rhs) = delete;
    public:
      virtual ProjectionPartition* as_partition_projection(void)
        { return this; }
      virtual bool is_disjoint(void) const;
#if 0
      virtual bool is_complete(void) const;
#endif
      virtual bool is_leaves_only(void) const;
      virtual bool is_unique_shards(void) const;
      virtual bool interferes(ProjectionNode *other, ShardID local) const;
      virtual void extract_shard_summaries(bool supports_name_based_analysis,
          ShardID local_shard, std::map<LogicalRegion,RegionSummary> &regions,
          std::map<LogicalPartition,PartitionSummary> &partitions) const;
      virtual void update_shard_summaries(bool supports_name_based_analysis,
          ShardID local_shard, size_t total_shards,
          std::map<LogicalRegion,RegionSummary> &regions,
          std::map<LogicalPartition,PartitionSummary> &partitions);
#if 0
      virtual RefinementNode* create_refinement(void) const;
#endif
      bool has_interference(ProjectionPartition *other, ShardID local) const;
      void add_child(ProjectionRegion *child);
#if 0
      void convert(const std::vector<ShardID> *shard_to_shard_mapping);
      void merge(ProjectionPartition *rhs,
                 const std::vector<ShardID> *shard_to_shard_mapping);
      void pack(Serializer &rez) const;
      void unpack(Deserializer &derez);
#endif
    public:
      PartitionNode *const partition;
      std::unordered_map<LegionColor,ProjectionRegion*> local_children;
      // This is only filled in if we support name-based dependence
      // analysis (disjoint and all users at the leaves)
      ShardedColorMap *name_based_children_shards;
    };

#if 0
    public:
      ProjectionTree(bool all_children_disjoint);
      ProjectionTree(const ProjectionTree &rhs) = delete;
      ~ProjectionTree(void);
    public:
      ProjectionTree& operator=(const ProjectionTree &rhs) = delete;
    public:
      bool interferes(const ProjectionTree *other, ShardID other_shard) const;
      bool uses_shard(ShardID other_shard) const;
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      std::map<LegionColor,ProjectionTree*> children;
      std::set<ShardID> users;
      const bool all_children_disjoint;
    };
#endif 

    /**
     * \class ProjectionSummary
     * A projection summary tracks the meta-data associated with a
     * particular projection including the projection tree that was
     * produced to represent it.
     */
    class ProjectionSummary : public Collectable {
    public:
      ProjectionSummary(const ProjectionInfo &info, ProjectionNode *node,
          const RegionRequirement &req, LogicalState *owner, bool dis,
          bool comp, bool permit_self, bool unique_shards);
      ProjectionSummary(const ProjectionSummary &rhs) = delete;
      ~ProjectionSummary(void);
    public:
      ProjectionSummary& operator=(const ProjectionSummary &rhs) = delete;
    public:
      bool matches(const ProjectionInfo &rhs,
                   const RegionRequirement &req) const;
    public:
      LogicalState *const owner;
      IndexSpaceNode *const domain;
      ProjectionFunction *const projection;
      ShardingFunction *const sharding;
      IndexSpaceNode *const sharding_domain;
      ProjectionNode *const result;
      const size_t arglen;
      void *const args;
      // We track a few different properties of this index space launch
      // that are useful for various different analyses and kinds of 
      // comparisons between index space launches
      // Whether we know all the points are disjoint from each other
      // based privileges of the projection and the projection function 
      const bool disjoint;
      // Whether this projection tree is complete or not according to
      // the projection functor
      const bool complete; 
      // Whether this projection summary can be analyzed against itself
      // using name-based dependence analysis which is that same as
      // having sub-regions described using a disjoint-only subtree
      // and all accesses at the leaves of the tree
      // Note that individual points in the same launch can still use
      // the same sub-regions here
      const bool permits_name_based_self_analysis;
      // Whether each region has a unique set of shards users
      const bool unique_shard_users;
    };

    /**
     * \class RefinementTracker
     * This class provides a generic interface for deciding when to
     * perform or change refinements of the region tree
     */
    class RefinementTracker {
    public:
      virtual ~RefinementTracker(void) { };
    public:
      virtual RegionRefinementTracker* as_region_tracker(void) { return NULL; }
      virtual PartitionRefinementTracker* as_partition_tracker(void)
        { return NULL; }
      virtual RefinementTracker* clone(void) const = 0;
      virtual void initialize_already_refined(void) = 0;
      virtual bool update_child(RegionTreeNode *child, 
                                const RegionUsage &usage,
                                bool &allow_refinement) = 0;
      virtual bool update_projection(ProjectionSummary *summary,
                                     const RegionUsage &usage,
                                     bool &allow_refinement) = 0;
      virtual bool update_arrival(const RegionUsage &usage,
                                  bool &allow_refinement) = 0;
      virtual void invalidate_refinement(ContextID ctx, 
                                const FieldMask &invalidation_mask) = 0;
#if 0
      virtual bool is_disjoint_complete(void) const = 0;
      virtual bool needs_fallback_refinement(
                                   const ProjectionInfo &proj_info) const = 0;
      virtual bool update_refinement_child(RegionTreeNode *child_node) = 0;
      virtual bool update_refinement_projection(
                                           ProjectionNode *projection) = 0;
      virtual void change_refinements(ContextID ctx, const FieldMask &mask,
                        FieldMaskSet<RegionTreeNode> &new_children,
                        FieldMaskSet<ProjectionNode> &new_projections) = 0;
      virtual void invalidate_refinement(ContextID ctx, 
                                      const FieldMask &invalidation_mask) = 0;
      virtual void find_child_refinements(
          std::set<RegionTreeNode*> &children) const = 0;
      virtual void convert(
          const std::vector<ShardID> *shard_to_shard_mapping) = 0;
      virtual void merge(RefinementTracker *rhs,
          const std::vector<ShardID> *shard_to_shard_mapping) = 0;
      virtual void pack(Serializer &rez, 
          std::map<LegionColor,RegionTreeNode*> &to_traverse) = 0;
#endif
    public:
      // This is the number of return children or projections we need
      // to observe in total before we consider a change to a refinement
      static constexpr uint64_t CHANGE_REFINEMENT_RETURN_COUNT = 256;
      // Check that this is a power of 2 for fast integer division
      static_assert((CHANGE_REFINEMENT_RETURN_COUNT & 
            (CHANGE_REFINEMENT_RETURN_COUNT - 1)) == 0, "must be power of two");
      // This is the weight for how degrading scores over time in our
      // exponentially weighted moving average computation
      static constexpr double CHANGE_REFINEMENT_RETURN_WEIGHT = 0.99;
      // This is the timeout for refinements where we will clear out all
      // candidate refinements and reset the state to look again
      static constexpr uint64_t CHANGE_REFINEMENT_TIMEOUT = 4096;
    protected:
      enum RefinementState {
        UNREFINED_STATE,
        COMPLETE_NONWRITE_REFINED_STATE,
        INCOMPLETE_NONWRITE_REFINED_STATE,
        COMPLETE_WRITE_REFINED_STATE,
        INCOMPLETE_WRITE_REFINED_STATE,
        NO_REFINEMENT_STATE,
      };
    };

    /**
     * \class RegionRefinementTracker
     * This class tracks the refinements (both partitions and projections)
     * on a region node in the region tree.
     */
    class RegionRefinementTracker : public RefinementTracker,
      public LegionHeapify<RegionRefinementTracker> {
    public:
      RegionRefinementTracker(RegionNode *node);
#if 0
      RegionRefinementTracker(RegionNode *node, bool current_refinement);
      RegionRefinementTracker(RegionNode *node, RegionTreeNode *child);
      RegionRefinementTracker(RegionNode *node, ProjectionRegion *projection);
#endif
      RegionRefinementTracker(const RegionRefinementTracker &rhs) = delete;
      virtual ~RegionRefinementTracker(void);
    public:
      RegionRefinementTracker& operator=(
                              const RegionRefinementTracker &rhs) = delete;
    public:
      virtual RegionRefinementTracker* as_region_tracker(void) { return this; }
      virtual RefinementTracker* clone(void) const;
      virtual void initialize_already_refined(void);
      virtual bool update_child(RegionTreeNode *child, 
                                const RegionUsage &usage,
                                bool &allow_refinement);
      virtual bool update_projection(ProjectionSummary *summary,
                                     const RegionUsage &usage,
                                     bool &allow_refinement);
      virtual bool update_arrival(const RegionUsage &usage,
                                  bool &allow_refinement);
      virtual void invalidate_refinement(ContextID ctx, 
                                         const FieldMask &invalidation_mask);
#if 0
      virtual bool is_disjoint_complete(void) const;
      virtual bool needs_fallback_refinement(
                                   const ProjectionInfo &proj_info) const;
      virtual bool update_refinement_child(RegionTreeNode *child_node);
      virtual bool update_refinement_projection(ProjectionNode *projection);
      virtual void change_refinements(ContextID ctx, const FieldMask &mask,
                        FieldMaskSet<RegionTreeNode> &new_children,
                        FieldMaskSet<ProjectionNode> &new_projections);
      virtual void invalidate_refinement(ContextID ctx, 
                                         const FieldMask &invalidation_mask);
      virtual void find_child_refinements(
          std::set<RegionTreeNode*> &children) const;
      virtual void convert(
          const std::vector<ShardID> *shard_to_shard_mapping);
      virtual void merge(RefinementTracker *rhs,
          const std::vector<ShardID> *shard_to_shard_mapping);
      virtual void pack(Serializer &rez, 
          std::map<LegionColor,RegionTreeNode*> &to_traverse);
      static RegionRefinementTracker* unpack(RegionNode *region,
          Deserializer &derez, std::map<LegionColor,RegionTreeNode*> &traverse);
#endif
    protected:
      bool is_dominant_candidate(double score, bool is_current);
      void invalidate_unused_candidates(void);
    public:
      RegionNode *const region;
#if 0
      // At most one of these will be non-NULL
      PartitionNode *const refined_child;
      ProjectionRegion *const refined_projection;
      // This region can be part of the refinement tree even if it
      // doesn't have a refined child or projection
      const bool current_refinement;
    
#endif
    protected: 
      RefinementState refinement_state;
      PartitionNode *refined_child;
      ProjectionRegion *refined_projection;
    protected:  
      // Track the candidate children and projections and how often
      // we have observed a return back to them
      // <current score,timestamp of last observed return>
      std::unordered_map<PartitionNode*,
        std::pair<double,uint64_t> > candidate_partitions;
      std::unordered_map<ProjectionRegion*,
        std::pair<double,uint64_t> > candidate_projections;
      // Monotonically increasing clock counting total number of traversals
      uint64_t total_traversals;
      // The timeout tracks how long we've gone without seeing a return
      // If we go for too long without seeing a return, we timeout and
      // clear out all the candidates so we can try again
      uint64_t return_timeout;  
    };

    /**
     * \class PartitionRefinementTracker
     * This class tracks the refinements (both sub-regions and projections)
     * on a region node in the region tree.
     */
    class PartitionRefinementTracker : public RefinementTracker,
      public LegionHeapify<PartitionRefinementTracker> {
    public:
      PartitionRefinementTracker(PartitionNode *node);
#if 0
      PartitionRefinementTracker(PartitionNode *node, bool current_refinement);
      PartitionRefinementTracker(PartitionNode *node,ProjectionPartition *proj);
#endif
      PartitionRefinementTracker(
          const PartitionRefinementTracker &rhs) = delete;
      virtual ~PartitionRefinementTracker(void);
    public:
      PartitionRefinementTracker& operator=(
          const PartitionRefinementTracker &rhs) = delete;
    public:
      virtual PartitionRefinementTracker* as_partition_tracker(void)
        { return this; }
      virtual RefinementTracker* clone(void) const;
      virtual void initialize_already_refined(void);
      virtual bool update_child(RegionTreeNode *child, 
                                const RegionUsage &usage,
                                bool &allow_refinement);
      virtual bool update_projection(ProjectionSummary *summary,
                                     const RegionUsage &usage,
                                     bool &allow_refinement);
      virtual bool update_arrival(const RegionUsage &usage,
                                  bool &allow_refinement);
      virtual void invalidate_refinement(ContextID ctx, 
                                         const FieldMask &invalidation_mask);
#if 0
      virtual bool is_disjoint_complete(void) const;
      virtual bool needs_fallback_refinement(
                                   const ProjectionInfo &proj_info) const;
      virtual bool update_refinement_child(RegionTreeNode *child_node);
      virtual bool update_refinement_projection(ProjectionNode *projection);
      virtual void change_refinements(ContextID ctx, const FieldMask &mask,
                        FieldMaskSet<RegionTreeNode> &new_children,
                        FieldMaskSet<ProjectionNode> &new_projections);
      virtual void invalidate_refinement(ContextID ctx, 
                                         const FieldMask &invalidation_mask);
      virtual void find_child_refinements(
          std::set<RegionTreeNode*> &children) const;
      virtual void convert(
          const std::vector<ShardID> *shard_to_shard_mapping);
      virtual void merge(RefinementTracker *rhs,
          const std::vector<ShardID> *shard_to_shard_mapping);
      virtual void pack(Serializer &rez, 
          std::map<LegionColor,RegionTreeNode*> &to_traverse);
      static PartitionRefinementTracker* unpack(PartitionNode *partition,
          Deserializer &derez, std::map<LegionColor,RegionTreeNode*> &traverse);
#endif
    protected:
      bool is_dominant_candidate(double score, bool is_current);
      void invalidate_unused_candidates(void);
    public:
      PartitionNode *const partition;
#if 0
      ProjectionPartition *const refined_projection;
      // A partition is only a current refinement if it has a
      // refined_projection or all its children are part of the refinement
      const bool current_refinement;
#endif
    protected:
      ProjectionPartition *refined_projection;
      RefinementState refinement_state;
      // These are children which are disjoint and complete
      // Note we don't need to hold references on them as they are kept alive
      // by the reference we are holding on their partition
      std::vector<RegionNode*> children;
      std::unordered_map<ProjectionPartition*,
        std::pair<double,uint64_t> > candidate_projections;
      // The individual score and last traversals of all the children
      double children_score;
      uint64_t children_last;
      // Monotonically increasing clock counting total number of traversals
      uint64_t total_traversals;
      // The timeout tracks how long we've gone without seeing a return
      // If we go for too long without seeing a return, we timeout and
      // clear out all the candidates so we can try again
      uint64_t return_timeout;
      // What fraction of children need to observed before we condider
      // this partition as being disjoint and complete, 1 means all the
      // children need to be observed to be considered complete, 2 means 
      // that half the children need to be observed before being complete,
      // 3 means a third need to be observed before being complete, etc
      static constexpr uint64_t CHANGE_REFINEMENT_PARTITION_FRACTION = 2;
    };

    /**
     * \class LogicalState
     * Track all the information about the current state
     * of a logical region from a given context. This
     * is effectively all the information at the analysis
     * wavefront for this particular logical region.
     */
    class LogicalState : public LegionHeapify<LogicalState> {
    public:
      static const AllocationType alloc_type = CURRENT_STATE_ALLOC;
    public:
      LogicalState(RegionTreeNode *owner, ContextID ctx);
      LogicalState(const LogicalState &state) = delete;
      ~LogicalState(void);
    public:
      LogicalState& operator=(const LogicalState &rhs) = delete;
    public:
      void check_init(void);
      void clear_logical_users(void);
      void reset(void);
      void clear_deleted_state(const FieldMask &deleted_mask);
#if 0
      void pack_refinements(Serializer &rez, 
                std::map<LegionColor,RegionTreeNode*> &to_traverse) const;
      void unpack_refinements(Deserializer &derez, 
                  std::map<LegionColor,RegionTreeNode*> &to_traverse);
      void merge_refinements(LogicalState &src,
          const std::vector<ShardID> *shard_to_shard_mapping,
          std::set<RegionTreeNode*> &to_traverse);
      void convert_refinements(LogicalState &src,
          const std::vector<ShardID> *shard_to_shard_mapping,
          std::set<RegionTreeNode*> &to_traverse);
#endif
      ProjectionSummary* find_or_create_projection_summary(
                                          Operation *op, unsigned index,
                                          const RegionRequirement &req,
                                          LogicalAnalysis &analysis,
                                          const ProjectionInfo &proj_info);
      void remove_projection_summary(ProjectionSummary *summary);
      bool has_interfering_shards(LogicalAnalysis &analysis,
                          ProjectionSummary *one, ProjectionSummary *two);
#if 0
      ProjectionNode* find_or_create_fallback_refinement(InnerContext *context,
                                                  IndexSpaceNode *color_space);
#endif
#ifdef DEBUG_LEGION
      void sanity_check(void) const;
#endif
    public:
      void initialize_refined_fields(const FieldMask &mask);
      void update_refinement_child(ContextID ctx, RegionTreeNode *child,
                                   const RegionUsage &usage,
                                   FieldMask &refinement_mask);
      void update_refinement_projection(ContextID ctx,
                                        ProjectionSummary *summary,
                                        const RegionUsage &usage,
                                        FieldMask &refinement_mask);
      void update_refinement_arrival(ContextID ctx, const RegionUsage &usage,
                                     FieldMask &refinement_mask);
      void invalidate_refinements(ContextID ctx, FieldMask invalidation_mask);
#if 0
      void initialize_unrefined_fields(const FieldMask &mask, 
          const unsigned index, LogicalAnalysis &analysis);
      void update_refinement_child(FieldMask &disjoint_complete_mask,
          FieldMask traversal_mask, RegionTreeNode *child_node, 
          FieldMask child_disjoint_complete,
          const ProjectionInfo &info, LogicalAnalysis &analysis,
          ContextID ctx, LogicalRegion privilege, unsigned req_index,
          FieldMaskSet<RefinementOp> &refinement_operations);
      void update_refinement_projection(FieldMask &disjoint_complete_mask,
          FieldMask traversal_mask, ProjectionSummary *projection,
          LogicalAnalysis &logical_analysis, ContextID ctx,
          LogicalRegion privilege, unsigned req_index,
          FieldMaskSet<RefinementOp> &refinement_operations);
      void change_refinements(ContextID ctx, size_t total_shards,
          FieldMask mask, FieldMaskSet<RefinementNode> &refinements);
      void invalidate_refinements(ContextID ctx, FieldMask invalidation_mask);
    public:
      bool find_symbolic_elide_close_result(const ProjectionSummary &prev, 
                            const ProjectionSummary &next, bool &result) const;
      void record_symbolic_elide_close_result(const ProjectionSummary &prev,
                            const ProjectionSummary &next, bool result);
#endif
    public:
      RegionTreeNode *const owner;
    public:
      LegionList<FieldState,LOGICAL_FIELD_STATE_ALLOC> field_states;
      // Note that even though these are field mask sets keyed on pointers
      // we mark them as determinsitic so that shards always iterate over
      // these elements in the same order
      typedef FieldMaskSet<LogicalUser,UNTRACKED_ALLOC,true/*determinisitic*/>
        OrderedFieldMaskUsers;
      OrderedFieldMaskUsers curr_epoch_users, prev_epoch_users; 
    public:
      // Refinement trackers manage the state of refinements for different
      // fields on this particular node of the region tree if we're along
      // a disjoint and complete partition path in this context.
      // We keep refinement trackers coalesced by fields as long as much
      // as possible but diverge them whenever unequal field sets try to
      // access them. They are grouped back together whenever we decide
      // to perform a refinement along this node.
      FieldMaskSet<RefinementTracker> refinement_trackers;
    public:
      static constexpr size_t PROJECTION_CACHE_SIZE = 32;
      // Note that this list can grow bigger than PROJECTION_CACHE_SIZE
      // but we only keep references on the entries within the size of the
      // cache. This allows us to still hit on projections that are still
      // alive from othe references, but also allows those entries to
      // be pruned out once they are no longer alive
      std::list<ProjectionSummary*> projection_summary_cache;
      std::unordered_map<ProjectionSummary*,
        std::unordered_map<ProjectionSummary*,bool> > interfering_shards;
#if 0
    public:
      // This is the fallback refinement for the logical region in this
      // context on this node of the region tree. Note that we only need
      // to compute this once and we can save it for the rest of the 
      // context. It will only be invalidated when this part of the
      // context is cleared out.
      ProjectionNode* fallback_refinement;
      // Track whether this node is part of the disjoint-complete tree
      FieldMask disjoint_complete_tree;
      // Use this data structure for tracking where the disjoint-complete
      // tree is for this region tree. On region nodes there should be at
      // most one child in this data structure. On partition nodes there
      // can be any number of children with different field masks.
      // Note that this might also be empty for partition nodes where
      // we have issued projections
      FieldMaskSet<RegionTreeNode> disjoint_complete_children;
      // Keep track of the disjoint complete accesses that have been
      // done in other children to track whether we want to change later
      // For partitions we'll only store the children to help with the
      // process of counting. After that we'll remove children and the
      // summary mask will be all that remains to record which fields
      // have disjoint and complete accesses
      FieldMaskSet<RegionTreeNode> disjoint_complete_accesses;
      // For partitions only, we record the counts of the numbers of
      // children that we've observed for all fields to see when we're 
      // close enough to be counted as being considered refined
      // For regions, we keep two counts, one of the number of
      // consecutive accesses to the most recent child in 
      // disjoint_complete_accesses (expressed as an even number 2*count)
      // and a second number the number of accesses to any child that
      // is not the current one in disjoint_complete_children
      // (expressed as an odd number 2*count+1)
      typedef LegionMap<size_t,FieldMask,UNTRACKED_ALLOC,
                        std::greater<size_t> > FieldSizeMap;
      FieldSizeMap                 disjoint_complete_child_counts;
      // If we have non-zero depth projection functions then we can get
      // these at the bottom of the disjoint complete access trees to say
      // how to project from a given node in the region tree
      FieldMaskSet<RefProjectionSummary> disjoint_complete_projections;
    public:
      struct SymbolicCacheEntry {
      public:
        SymbolicCacheEntry(const ProjectionSummary o,
                           const ProjectionSummary t, bool r)
          : one(o), two(t), result(r) { }
      public:
        inline bool matches(const ProjectionSummary &prev, 
                            const ProjectionSummary &next) const
        {
          if (one != prev) return false;
          if (two != next) return false;
          return true;
        }
      public:
        ProjectionSummary one, two;
        bool result;
      };
      // This helps to memoize expensive close operation elisions tests 
      // within this context in a determinstic way for control replication
      std::list<SymbolicCacheEntry> *symbolic_elide_close_results;
#endif
    };

    typedef DynamicTableAllocator<LogicalState,10,8> LogicalStateAllocator;

#if 0
    /**
     * \class LogicalCloser
     * This structure helps keep track of the state
     * necessary for performing a close operation
     * on the logical region tree.
     */
    class LogicalCloser {
    public:
      LogicalCloser(ContextID ctx, const LogicalUser &u, 
                    RegionTreeNode *root, bool validates);
      LogicalCloser(const LogicalCloser &rhs) = delete;
      ~LogicalCloser(void);
    public:
      LogicalCloser& operator=(const LogicalCloser &rhs) = delete;
    public:
      inline bool has_close_operations(FieldMask &already_closed_mask)
        {
          if (!close_mask)
            return false;
          if (!!already_closed_mask)
          {
            // Remove any fields which were already closed
            // We only need one close per field for a traversal
            // This handles the upgrade cases after we've already
            // done a closer higher up in the tree
            close_mask -= already_closed_mask;
            if (!close_mask)
              return false;
          }
          already_closed_mask |= close_mask;
          return true;
        }
      // Record normal closes like this
      void record_close_operation(const FieldMask &mask);
      void record_closed_user(const LogicalUser &user, const FieldMask &mask);
#ifndef LEGION_SPY
      void pop_closed_user(void);
#endif
      void initialize_close_operations(LogicalState &state, 
                                       Operation *creator,
                                       const LogicalTraceInfo &trace_info,
                                       const bool check_for_refinements,
                                       const bool has_next_child);
      void perform_dependence_analysis(const LogicalUser &current,
                                       const FieldMask &open_below,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC> &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC> &pusers);
      void update_state(LogicalState &state);
      void register_close_operations(
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC> &users);
    protected:
      void register_dependences(CloseOp *close_op, 
                                const LogicalUser &close_user,
                                const LogicalUser &current, 
                                const FieldMask &open_below,
             LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC> &husers,
             LegionList<LogicalUser,LOGICAL_REC_ALLOC> &ausers,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC> &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC> &pusers);
    public:
      const ContextID ctx;
      const LogicalUser &user;
      RegionTreeNode *const root_node;
      const bool validates;
      const bool tracing;
      LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC> closed_users;
    protected:
      FieldMask close_mask;
    protected:
      // At most we will ever generate one close operation at a node
      MergeCloseOp *close_op;
    protected:
      // Cache the generation IDs so we can kick off ops before adding users
      GenerationID merge_close_gen;
    }; 
#endif

    /**
     * \class LogicalAnalysis 
     * The logical analysis helps capture the state of region tree traversals
     * for all region requirements in an operation. This includes capturing
     * the needed refinements to be performed as well as closes that have
     * already been performed for various projection region requirements.
     * At the end of the analysis, it issues all the refinements to be performed
     * by an operation and then performs them after all of the region 
     * requirements for that operation are done being analyzed. That ensures
     * that we have at most one refinement change for all region requirements
     * in an operation that touch the same fields of the same region tree.
     */
    class LogicalAnalysis {
    public:
      struct PendingClose : public LegionHeapify<PendingClose> {
      public:
        PendingClose(RegionTreeNode *n, unsigned idx)
          : node(n), req_idx(idx) { }
      public:
        FieldMaskSet<LogicalUser> preconditions;
        RegionTreeNode *const node;
        const unsigned req_idx;
      };
    public:
      LogicalAnalysis(Operation *op);
      LogicalAnalysis(const LogicalAnalysis &rhs) = delete;
      ~LogicalAnalysis(void);
    public:
      LogicalAnalysis& operator=(const LogicalAnalysis &rhs) = delete;
    public:
#if 0
      RefinementOp* create_refinement(const LogicalUser &user,
          PartitionNode *partition, const FieldMask &refinement_mask,
          LogicalRegion privilege_root);
      bool deduplicate(PartitionNode *child, FieldMask &refinement_mask);
#endif
      void record_unrefined_fields(RegionNode *node, unsigned index,
                                   const FieldMask &unrefined);
      void record_pending_refinement(LogicalRegion privilege,
                                     unsigned req_index,
                                     unsigned parent_req_index,
                                     RegionTreeNode *refinement_node,
                                     FieldMask refinement_mask,
                                     FieldMaskSet<RefinementOp> &refinements);
    public:
      // Record a prior operation that we need to depend on with a 
      // close operation to group together dependences
      void record_close_dependence(LogicalRegion privilege,
                                   unsigned req_index,
                                   RegionTreeNode *path_node,
                                   const LogicalUser *user,
                                   const FieldMask &mask);
    protected:
      void issue_internal_operation(RegionTreeNode *node,
          InternalOp *close_op, const FieldMask &internal_mask) const;
    public:
      Operation *const op;
      InnerContext *const context;
    protected:
#if 0
      FieldMaskSet<RegionNode> unrefined_nodes;
      std::map<RegionNode*,unsigned> unrefined_indexes;
#endif
      // Keep these ordered by the order in which we make them so that
      // all shards will iterate over them in the same order for 
      // control replication cases, we do this by sorting them based
      // on their unique IDs which are monotonically increasing so we
      // know that they will be in order across shards too
      typedef FieldMaskSet<RefinementOp,UNTRACKED_ALLOC,true/*ordered*/>
        OrderedRefinements;
      OrderedRefinements pending_refinements;
      std::map<RegionTreeNode*,MergeCloseOp*> pending_closes;
    };

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef : public LegionHeapify<InstanceRef> {
    public:
      InstanceRef(bool composite = false);
      InstanceRef(const InstanceRef &rhs);
      InstanceRef(InstanceManager *manager, const FieldMask &valid_fields);
      ~InstanceRef(void);
    public:
      InstanceRef& operator=(const InstanceRef &rhs);
    public:
      bool operator==(const InstanceRef &rhs) const;
      bool operator!=(const InstanceRef &rhs) const;
    public:
      inline bool has_ref(void) const { return (manager != NULL); }
      inline InstanceManager* get_manager(void) const { return manager; }
      inline const FieldMask& get_valid_fields(void) const 
        { return valid_fields; }
      inline void update_fields(const FieldMask &update) 
        { valid_fields |= update; }
    public:
      inline bool is_local(void) const { return local; }
      MappingInstance get_mapping_instance(void) const;
      bool is_virtual_ref(void) const; 
    public:
      void add_resource_reference(ReferenceSource source) const;
      void remove_resource_reference(ReferenceSource source) const;
      bool acquire_valid_reference(ReferenceSource source) const;
      void add_valid_reference(ReferenceSource source) const; 
      void remove_valid_reference(ReferenceSource source) const;
    public:
      Memory get_memory(void) const;
      PhysicalManager* get_physical_manager(void) const;
    public:
      bool is_field_set(FieldID fid) const;
      LegionRuntime::Accessor::RegionAccessor<
          LegionRuntime::Accessor::AccessorType::Generic>
            get_accessor(void) const;
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      void pack_reference(Serializer &rez) const;
      void unpack_reference(Runtime *rt, Deserializer &derez, RtEvent &ready);
    protected:
      FieldMask valid_fields; 
      InstanceManager *manager;
      bool local;
    };

    /**
     * \class InstanceSet
     * This class is an abstraction for representing one or more
     * instance references. It is designed to be light-weight and
     * easy to copy by value. It maintains an internal copy-on-write
     * data structure to avoid unnecessary premature copies.
     */
    class InstanceSet {
    public:
      struct CollectableRef : public Collectable, public InstanceRef {
      public:
        CollectableRef(void)
          : Collectable(), InstanceRef() { }
        CollectableRef(const InstanceRef &ref)
          : Collectable(), InstanceRef(ref) { }
        CollectableRef(const CollectableRef &rhs)
          : Collectable(), InstanceRef(rhs) { }
        ~CollectableRef(void) { }
      public:
        CollectableRef& operator=(const CollectableRef &rhs);
      };
      struct InternalSet : public Collectable {
      public:
        InternalSet(size_t size = 0)
          { if (size > 0) vector.resize(size); }
        InternalSet(const InternalSet &rhs) : vector(rhs.vector) { }
        ~InternalSet(void) { }
      public:
        InternalSet& operator=(const InternalSet &rhs)
          { assert(false); return *this; }
      public:
        inline bool empty(void) const { return vector.empty(); }
      public:
        LegionVector<InstanceRef> vector; 
      };
    public:
      InstanceSet(size_t init_size = 0);
      InstanceSet(const InstanceSet &rhs);
      ~InstanceSet(void);
    public:
      InstanceSet& operator=(const InstanceSet &rhs);
      bool operator==(const InstanceSet &rhs) const;
      bool operator!=(const InstanceSet &rhs) const;
    public:
      InstanceRef& operator[](unsigned idx);
      const InstanceRef& operator[](unsigned idx) const;
    public:
      bool empty(void) const;
      size_t size(void) const;
      void resize(size_t new_size);
      void clear(void);
      void swap(InstanceSet &rhs);
      void add_instance(const InstanceRef &ref);
      bool is_virtual_mapping(void) const;
    public:
      void pack_references(Serializer &rez) const;
      void unpack_references(Runtime *runtime, Deserializer &derez, 
                             std::set<RtEvent> &ready_events);
    public:
      void add_resource_references(ReferenceSource source) const;
      void remove_resource_references(ReferenceSource source) const;
      bool acquire_valid_references(ReferenceSource source) const;
      void add_valid_references(ReferenceSource source) const;
      void remove_valid_references(ReferenceSource source) const;
    public:
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    protected:
      void make_copy(void);
    protected:
      union {
        CollectableRef* single;
        InternalSet*     multi;
      } refs;
      bool single;
      mutable bool shared;
    };

    /**
     * \class CopyFillGuard
     * This is the base class for copy fill guards. It serves as a way to
     * ensure that multiple readers that can race to update an equivalence
     * set observe each others changes before performing their copies.
     */
    class CopyFillGuard {
    private:
      struct CopyFillDeletion : public LgTaskArgs<CopyFillDeletion> {
      public:
        static const LgTaskID TASK_ID = LG_COPY_FILL_DELETION_TASK_ID;
      public:
        CopyFillDeletion(CopyFillGuard *g, UniqueID uid, RtUserEvent r)
          : LgTaskArgs<CopyFillDeletion>(uid), guard(g), released(r) { }
      public:
        CopyFillGuard *const guard;
        const RtUserEvent released;
      };
    public:
#ifndef NON_AGGRESSIVE_AGGREGATORS
      CopyFillGuard(RtUserEvent post, RtUserEvent applied);
#else
      CopyFillGuard(RtUserEvent applied);
#endif
      CopyFillGuard(const CopyFillGuard &rhs);
      virtual ~CopyFillGuard(void);
    public:
      CopyFillGuard& operator=(const CopyFillGuard &rhs);
    public:
      void pack_guard(Serializer &rez);
      static CopyFillGuard* unpack_guard(Deserializer &derez, Runtime *rt,
                                         EquivalenceSet *set);
    public:
      bool record_guard_set(EquivalenceSet *set, bool read_only_guard);
      bool release_guards(Runtime *runtime, std::set<RtEvent> &applied,
                          bool force_deferral = false);
      static void handle_deletion(const void *args); 
    private:
      void release_guarded_sets(std::set<RtEvent> &released);
    public:
#ifndef NON_AGGRESSIVE_AGGREGATORS
      const RtUserEvent guard_postcondition;
#endif
      const RtUserEvent effects_applied;
    private:
      mutable LocalLock guard_lock;     
      // Record equivalence classes for which we need to remove guards
      std::set<EquivalenceSet*> guarded_sets;
      // Keep track of any events for remote releases
      std::vector<RtEvent> remote_release_events;
      // Track whether we are releasing or not
      bool releasing_guards;
      // Track whether this is a read-only guard or not
      bool read_only_guard;
    };

    /**
     * \class CopyFillAggregator
     * The copy aggregator class is one that records the copies
     * that needs to be done for different equivalence classes and
     * then merges them together into the biggest possible copies
     * that can be issued together.
     */
    class CopyFillAggregator : public CopyFillGuard,
                               public LegionHeapify<CopyFillAggregator> {
    public:
      static const AllocationType alloc_type = COPY_FILL_AGGREGATOR_ALLOC;
    public:
      struct CopyFillAggregation : public LgTaskArgs<CopyFillAggregation>,
                                   public PhysicalTraceInfo {
      public:
        static const LgTaskID TASK_ID = LG_COPY_FILL_AGGREGATION_TASK_ID;
      public:
        CopyFillAggregation(CopyFillAggregator *a, const PhysicalTraceInfo &i,
                            ApEvent p, const bool manage_dst, 
                            const bool restricted, UniqueID uid,
                            std::map<InstanceView*,std::vector<ApEvent> > *dsts)
          : LgTaskArgs<CopyFillAggregation>(uid), PhysicalTraceInfo(i),
            dst_events((dsts == NULL) ? NULL : 
                new std::map<InstanceView*,std::vector<ApEvent> >()),
            aggregator(a), pre(p), manage_dst_events(manage_dst),
            restricted_output(restricted)
          // This is kind of scary, Realm is about to make a copy of this
          // without our knowledge, but we need to preserve the correctness
          // of reference counting on PhysicalTraceRecorders, so just add
          // an extra reference here that we will remove when we're handled.
          { if (rec != NULL) rec->add_recorder_reference(); 
            if (dsts != NULL) dst_events->swap(*dsts); }
      public:
        inline void remove_recorder_reference(void) const
          { if ((rec != NULL) && rec->remove_recorder_reference()) delete rec; }
      public:
        std::map<InstanceView*,std::vector<ApEvent> > *const dst_events;
        CopyFillAggregator *const aggregator;
        const ApEvent pre;
        const bool manage_dst_events;
        const bool restricted_output;
      }; 
    public:
      typedef LegionMap<InstanceView*,
               FieldMaskSet<IndexSpaceExpression> > InstanceFieldExprs;
      typedef LegionMap<ApEvent,FieldMask> EventFieldMap;
      class CopyUpdate;
      class FillUpdate;
      class Update {
      public:
        Update(IndexSpaceExpression *exp, const FieldMask &mask,
               CopyAcrossHelper *helper);
        virtual ~Update(void); 
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const = 0;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills) = 0;
      public:
        IndexSpaceExpression *const expr;
        const FieldMask src_mask;
        CopyAcrossHelper *const across_helper;
      };
      class CopyUpdate : public Update, public LegionHeapify<CopyUpdate> {
      public:
        CopyUpdate(InstanceView *src, PhysicalManager *man,
                   const FieldMask &mask,
                   IndexSpaceExpression *expr,
                   ReductionOpID red = 0,
                   CopyAcrossHelper *helper = NULL)
          : Update(expr, mask, helper), 
            source(src), src_man(man), redop(red) { }
        virtual ~CopyUpdate(void) { }
      private:
        CopyUpdate(const CopyUpdate &rhs) = delete;
        CopyUpdate& operator=(const CopyUpdate &rhs) = delete;
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        InstanceView *const source;
        PhysicalManager *const src_man; // which source manager for collectives
        const ReductionOpID redop;
      };
      class FillUpdate : public Update, public LegionHeapify<FillUpdate> {
      public:
        FillUpdate(FillView *src, const FieldMask &mask,
                   IndexSpaceExpression *expr, PredEvent guard,
                   CopyAcrossHelper *helper = NULL)
          : Update(expr, mask, helper), source(src), fill_guard(guard) { }
        virtual ~FillUpdate(void) { }
      private:
        FillUpdate(const FillUpdate &rhs) = delete;
        FillUpdate& operator=(const FillUpdate &rhs) = delete;
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        FillView *const source;
        // Unlike copies, because of nested predicated fill operations,
        // then fills can have their own predication guard different
        // from the base predicate guard for the aggregator
        const PredEvent fill_guard;
      };
      typedef LegionMap<ApEvent,FieldMaskSet<Update> > EventFieldUpdates;
    public:
      CopyFillAggregator(RegionTreeForest *forest, PhysicalAnalysis *analysis,
                         CopyFillGuard *previous, bool track_events,
                         PredEvent pred_guard = PredEvent::NO_PRED_EVENT);
      CopyFillAggregator(RegionTreeForest *forest, PhysicalAnalysis *analysis,
                         unsigned src_index, unsigned dst_idx,
                         CopyFillGuard *previous, bool track_events,
                         PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
                         // Used only in the case of copy-across analyses
                         RtEvent alternate_pre = RtEvent::NO_RT_EVENT);
      CopyFillAggregator(const CopyFillAggregator &rhs) = delete;
      virtual ~CopyFillAggregator(void);
    public:
      CopyFillAggregator& operator=(const CopyFillAggregator &rhs) = delete;
    public:
      void record_update(InstanceView *dst_view,
                          PhysicalManager *dst_man,
                          LogicalView *src_view,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          const PhysicalTraceInfo &trace_info,
                          EquivalenceSet *tracing_eq,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL); 
      void record_updates(InstanceView *dst_view, 
                          PhysicalManager *dst_man,
                          const FieldMaskSet<LogicalView> &src_views,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          const PhysicalTraceInfo &trace_info,
                          EquivalenceSet *tracing_eq,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      void record_partial_updates(InstanceView *dst_view,
                          PhysicalManager *dst_man,
                          const LegionMap<LogicalView*,
                          FieldMaskSet<IndexSpaceExpression> > &src_views,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          const PhysicalTraceInfo &trace_info,
                          EquivalenceSet *tracing_eq,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      // Neither fills nor reductions should have a redop across as they
      // should have been applied an instance directly for across copies
      void record_fill(InstanceView *dst_view,
                       FillView *src_view,
                       const FieldMask &fill_mask,
                       IndexSpaceExpression *expr,
                       const PredEvent fill_guard,
                       EquivalenceSet *tracing_eq,
                       CopyAcrossHelper *across_helper = NULL);
      void record_reductions(InstanceView *dst_view,
                             PhysicalManager *dst_man,
                             const std::list<std::pair<InstanceView*,
                                    IndexSpaceExpression*> > &src_views,
                             const unsigned src_fidx,
                             const unsigned dst_fidx,
                             EquivalenceSet *tracing_eq,
                             CopyAcrossHelper *across_helper = NULL);
      ApEvent issue_updates(const PhysicalTraceInfo &trace_info, 
                            ApEvent precondition,
                            const bool restricted_output = false,
                            // Next args are used for across-copies
                            // to indicate that the precondition already
                            // describes the precondition for the 
                            // destination instance
                            const bool manage_dst_events = true,
                            std::map<InstanceView*,
                                     std::vector<ApEvent> > *dst_events = NULL);
    protected:
      void record_view(LogicalView *new_view);
      void resize_reductions(size_t new_size);
      void update_tracing_valid_views(EquivalenceSet *tracing_eq,
            LogicalView *src, LogicalView *dst, const FieldMask &mask,
            IndexSpaceExpression *expr, ReductionOpID redop) const;
      void record_instance_update(InstanceView *dst_view,
                          InstanceView *src_view,
                          PhysicalManager *src_man,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          EquivalenceSet *tracing_eq,
                          ReductionOpID redop,
                          CopyAcrossHelper *across_helper);
      struct SelectSourcesResult;
      const SelectSourcesResult& select_sources(InstanceView *target,
                          PhysicalManager *target_manager,
                          const std::vector<InstanceView*> &sources);
      void perform_updates(const LegionMap<InstanceView*,
                            FieldMaskSet<Update> > &updates,
                           const PhysicalTraceInfo &trace_info,
                           const ApEvent all_precondition, 
                           std::set<RtEvent> &recorded_events, 
                           const int redop_index,
                           const bool manage_dst_events,
                           const bool restricted_output,
                           std::map<InstanceView*,
                                    std::vector<ApEvent> > *dst_events);
      void issue_fills(InstanceView *target,
                       const std::vector<FillUpdate*> &fills,
                       std::set<RtEvent> &recorded_events,
                       const ApEvent precondition, const FieldMask &fill_mask,
                       const PhysicalTraceInfo &trace_info,
                       const bool manage_dst_events,
                       const bool restricted_output,
                       std::vector<ApEvent> *dst_events);
      void issue_copies(InstanceView *target, 
                        std::map<InstanceView*,
                                 std::vector<CopyUpdate*> > &copies,
                        std::set<RtEvent> &recorded_events,
                        const ApEvent precondition, const FieldMask &copy_mask,
                        const PhysicalTraceInfo &trace_info,
                        const bool manage_dst_events,
                        const bool restricted_output,
                        std::vector<ApEvent> *dst_events);
    public:
      inline void clear_update_fields(void) 
        { update_fields.clear(); } 
      inline bool has_update_fields(void) const 
        { return !!update_fields; }
      inline const FieldMask& get_update_fields(void) const 
        { return update_fields; }
    public:
      static void handle_aggregation(const void *args); 
    public:
      RegionTreeForest *const forest;
      const AddressSpaceID local_space;
      PhysicalAnalysis *const analysis;
      const unsigned src_index;
      const unsigned dst_index;
      const RtEvent guard_precondition;
      const PredEvent predicate_guard;
      const bool track_events;
    protected:
      FieldMask update_fields;
      LegionMap<InstanceView*,FieldMaskSet<Update> > sources; 
      std::vector</*vector over reduction epochs*/
        LegionMap<InstanceView*,FieldMaskSet<Update> > > reductions;
      // Figure out the reduction operator is for each epoch of a
      // given destination instance and field
      std::map<std::pair<InstanceView*,unsigned/*dst fidx*/>,
               std::vector<ReductionOpID> > reduction_epochs;
      std::set<LogicalView*> all_views; // used for reference counting
    protected:
      // Runtime mapping effects that we create
      std::set<RtEvent> effects; 
      // Events for the completion of our copies if we are supposed
      // to be tracking them
      std::vector<ApEvent> events;
      // An event to represent the merge of all the events above
      ApUserEvent summary_event;
    protected:
      struct SelectSourcesResult {
      public:
        SelectSourcesResult(void) { }
        SelectSourcesResult(std::vector<InstanceView*> &&srcs,
                            std::vector<unsigned> &&rank,
                            std::map<unsigned,PhysicalManager*> &&pts)
          : sources(srcs), ranking(rank), points(pts) { }
      public:
        inline bool matches(const std::vector<InstanceView*> &srcs) const
          {
            if (srcs.size() != sources.size())
              return false;
            for (unsigned idx = 0; idx < sources.size(); idx++)
              if (srcs[idx] != sources[idx])
                return false;
            return true;
          }
      public:
        std::vector<InstanceView*> sources;
        std::vector<unsigned> ranking;
        std::map<unsigned,PhysicalManager*> points;
      };
      // Cached calls to the mapper for selecting sources
      std::map<std::pair<InstanceView*,PhysicalManager*>,
               std::vector<SelectSourcesResult> > mapper_queries;
    };

    /**
     * \class PhysicalAnalysis
     * This is the virtual base class for handling all traversals over 
     * equivalence set trees to perform physical analyses
     */
    class PhysicalAnalysis : public Collectable, public LocalLock {
    private:
      struct DeferPerformTraversalArgs :
        public LgTaskArgs<DeferPerformTraversalArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_TRAVERSAL_TASK_ID;
      public:
        DeferPerformTraversalArgs(PhysicalAnalysis *ana,
                                  const VersionInfo &info);
      public:
        PhysicalAnalysis *const analysis;
        const VersionInfo *const version_info;
        RtUserEvent done_event;
      };
      struct DeferPerformAnalysisArgs :
        public LgTaskArgs<DeferPerformAnalysisArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_ANALYSIS_TASK_ID;
      public:
        DeferPerformAnalysisArgs(PhysicalAnalysis *ana, EquivalenceSet *set,
         const FieldMask &mask, RtUserEvent done, bool already_deferred = true);
      public:
        PhysicalAnalysis *const analysis;
        EquivalenceSet *const set;
        FieldMask *const mask;
        const RtUserEvent done_event;
        const bool already_deferred;
      };
      struct DeferPerformRemoteArgs : 
        public LgTaskArgs<DeferPerformRemoteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_REMOTE_TASK_ID;
      public:
        DeferPerformRemoteArgs(PhysicalAnalysis *ana); 
      public:
        PhysicalAnalysis *const analysis;
        const RtUserEvent done_event;
      };
      struct DeferPerformUpdateArgs : 
        public LgTaskArgs<DeferPerformUpdateArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_UPDATE_TASK_ID;
      public:
        DeferPerformUpdateArgs(PhysicalAnalysis *ana);
      public:
        PhysicalAnalysis *const analysis;
        const RtUserEvent done_event;
      };
      struct DeferPerformRegistrationArgs :
        public LgTaskArgs<DeferPerformRegistrationArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_REGISTRATION_TASK_ID;
      public:
        DeferPerformRegistrationArgs(PhysicalAnalysis *ana,
            const RegionUsage &use, const PhysicalTraceInfo &trace_info,
            ApEvent init_precondition, ApEvent termination, bool symbolic);
      public:
        PhysicalAnalysis *const analysis;
        const RegionUsage usage;
        const PhysicalTraceInfo *trace_info;
        const ApEvent precondition;
        const ApEvent termination;
        const ApUserEvent instances_ready;
        const RtUserEvent done_event;
        const bool symbolic;
      };
      struct DeferPerformOutputArgs : 
        public LgTaskArgs<DeferPerformOutputArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_OUTPUT_TASK_ID;
      public:
        DeferPerformOutputArgs(PhysicalAnalysis *ana, bool track,
                               const PhysicalTraceInfo &trace_info);
      public:
        PhysicalAnalysis *const analysis;
        const PhysicalTraceInfo *trace_info;
        const ApUserEvent effects_event;
      };
    public:
      // Local physical analysis
      PhysicalAnalysis(Runtime *rt, Operation *op, unsigned index,
                       IndexSpaceExpression *expr, bool on_heap,
                       bool immutable, bool exclusive = false,
                       CollectiveMapping *mapping = NULL,
                       bool first_local = true);
      // Remote physical analysis
      PhysicalAnalysis(Runtime *rt, AddressSpaceID source, AddressSpaceID prev,
                       Operation *op, unsigned index, 
                       IndexSpaceExpression *expr, bool on_heap,
                       bool immutable = false,CollectiveMapping *mapping = NULL,
                       bool exclusive = false, bool first_local = true);
      PhysicalAnalysis(const PhysicalAnalysis &rhs) = delete;
      virtual ~PhysicalAnalysis(void);
    public:
      inline bool has_remote_sets(void) const
        { return !remote_sets.empty(); }
      inline void record_parallel_traversals(void)
        { parallel_traversals = true; } 
      inline bool is_replicated(void) const 
        { return (collective_mapping != NULL); }
      inline CollectiveMapping* get_replicated_mapping(void) const
        { return collective_mapping; }
      inline bool is_collective_first_local(void) const
        { return collective_first_local; }
    public:
      void analyze(EquivalenceSet *set, const FieldMask &mask, 
                   std::set<RtEvent> &deferral_events,
                   std::set<RtEvent> &applied_events,
                   RtEvent precondition = RtEvent::NO_RT_EVENT,
                   const bool already_deferred = false);
      RtEvent defer_traversal(RtEvent precondition, const VersionInfo &info,
                              std::set<RtEvent> &applied_events);
      void defer_analysis(RtEvent precondition, EquivalenceSet *set,
              const FieldMask &mask, std::set<RtEvent> &deferral_events, 
              std::set<RtEvent> &applied_events,
              RtUserEvent deferral_event = RtUserEvent::NO_RT_USER_EVENT,
              const bool already_deferred = true);
      RtEvent defer_remote(RtEvent precondition, std::set<RtEvent> &applied);
      RtEvent defer_updates(RtEvent precondition, std::set<RtEvent> &applied);
      RtEvent defer_registration(RtEvent precondition,
                                 const RegionUsage &usage,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info,
                                 ApEvent init_precondition,
                                 ApEvent termination_event,
                                 ApEvent &instances_ready,
                                 bool symbolic);
      ApEvent defer_output(RtEvent precondition, const PhysicalTraceInfo &info,
                           bool track, std::set<RtEvent> &applied_events);
      void record_deferred_applied_events(std::set<RtEvent> &applied);
    public:
      virtual RtEvent perform_traversal(RtEvent precondition,
                                        const VersionInfo &version_info,
                                        std::set<RtEvent> &applied_events);
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false) = 0;
      virtual RtEvent perform_remote(RtEvent precondition, 
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
      virtual RtEvent perform_registration(RtEvent precondition,
                                           const RegionUsage &usage,
                                           std::set<RtEvent> &applied_events,
                                           ApEvent init_precondition,
                                           ApEvent termination_event,
                                           ApEvent &instances_ready,
                                           bool symbolic = false);
      virtual ApEvent perform_output(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual IndexSpaceID get_collective_match_space(void) const { return 0; }
    public:
      void process_remote_instances(Deserializer &derez,
                                    std::set<RtEvent> &ready_events);
      void process_local_instances(const FieldMaskSet<LogicalView> &views,
                                   const bool local_restricted);
      void filter_remote_expressions(FieldMaskSet<IndexSpaceExpression> &exprs);
      // Return true if any are restricted
      bool report_instances(FieldMaskSet<LogicalView> &instances);
    public:
      // Lock taken by these methods if needed
      void record_remote(EquivalenceSet *set, const FieldMask &mask, 
                         const AddressSpaceID owner);
    public:
      // Lock must be held from caller
      void record_instance(LogicalView* view, const FieldMask &mask);
      inline void record_restriction(void) { restricted = true; }
    public:
      static void handle_remote_instances(Deserializer &derez, Runtime *rt);
      static void handle_deferred_traversal(const void *args);
      static void handle_deferred_analysis(const void *args);
      static void handle_deferred_remote(const void *args);
      static void handle_deferred_update(const void *args);
      static void handle_deferred_registration(const void *args);
      static void handle_deferred_output(const void *args);
    public:
      const AddressSpaceID previous;
      const AddressSpaceID original_source;
      Runtime *const runtime;
      IndexSpaceExpression *const analysis_expr;
    public:
      Operation *const op;
      const unsigned index;
      const bool owns_op;
      const bool on_heap;
      // whether this is an exclusive analysis (e.g. read-write privileges)
      const bool exclusive; 
      // whether this is an immutable analysis (e.g. only reading eq set)
      const bool immutable;
    protected:
      // whether this is the first collective analysis on this address space
      bool collective_first_local;
    private:
      // This tracks whether this analysis is being used 
      // for parallel traversals or not
      bool parallel_traversals;
    protected: 
      bool restricted;
      LegionMap<AddressSpaceID,
                FieldMaskSet<EquivalenceSet> > remote_sets; 
      FieldMaskSet<LogicalView> *recorded_instances;
    protected:
      CollectiveMapping *collective_mapping;
    private:
      // We don't want to make a separate "applied" event for every single
      // deferred task that we need to launch. Therefore, we just make one
      // for each analysis on the first deferred task that is launched for
      // an analysis and record it. We then accumulate all the applied 
      // events for all the deferred stages and when the analysis is deleted
      // then we close the loop and merge all the accumulated events
      // into the deferred_applied_event
      RtUserEvent deferred_applied_event;
      std::set<RtEvent> deferred_applied_events;
    };

    /**
     * \class RegistrationAnalysis
     * A registration analysis is a kind of physical analysis that
     * also supports performing registration on the views of the 
     */
    class RegistrationAnalysis : public PhysicalAnalysis {
    public:
      RegistrationAnalysis(Runtime *rt, Operation *op, unsigned index,
                           RegionNode *node, bool on_heap,
                           const PhysicalTraceInfo &trace_info, bool exclusive);
      RegistrationAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                           Operation *op, unsigned index, 
                           RegionNode *node, bool on_heap, 
                           std::vector<PhysicalManager*> &&target_insts,
                           LegionVector<
                              FieldMaskSet<InstanceView> > &&target_views,
                           std::vector<IndividualView*> &&source_views,
                           const PhysicalTraceInfo &trace_info,
                           CollectiveMapping *collective_mapping,
                           bool first_local, bool exclusive);
      // Remote registration analysis with no views
      RegistrationAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                           Operation *op, unsigned index, 
                           RegionNode *node, bool on_heap, 
                           const PhysicalTraceInfo &trace_info,
                           CollectiveMapping *collective_mapping,
                           bool first_local, bool exclusive);
    public:
      virtual ~RegistrationAnalysis(void);
    public:
      RtEvent convert_views(LogicalRegion region, const InstanceSet &targets,
          const std::vector<PhysicalManager*> *sources = NULL,
          const RegionUsage *usage = NULL, bool collective_rendezvous = false,
          unsigned analysis_index = 0);
    public:
      virtual RtEvent perform_registration(RtEvent precondition,
                                           const RegionUsage &usage,
                                           std::set<RtEvent> &applied_events,
                                           ApEvent init_precondition,
                                           ApEvent termination_event,
                                           ApEvent &instances_ready,
                                           bool symbolic = false);
      virtual IndexSpaceID get_collective_match_space(void) const; 
    public:
      RegionNode *const region;
      const size_t context_index;
      const PhysicalTraceInfo trace_info;
    public:
      // Be careful to only access these after they are ready
      std::vector<PhysicalManager*> target_instances;
      LegionVector<FieldMaskSet<InstanceView> > target_views;
      std::map<InstanceView*,size_t> collective_arrivals;
      std::vector<IndividualView*> source_views;
    };

    /**
     * \class CollectiveAnalysis
     * This base class provides a virtual interface for collective
     * analyses. Collective analyses are registered with the collective
     * views to help with performing collective copy operations since
     * control for collective copy operations originates from just a
     * single physical analysis. We don't want to attribute all those
     * copies and trace recordings to a single analysis, so instead
     * we register CollectiveAnalysis objects with collective views
     * so that they can be used to help issue the copies.
     */
    class CollectiveAnalysis {
    public:
      virtual ~CollectiveAnalysis(void) { }
      virtual size_t get_context_index(void) const = 0;
      virtual unsigned get_requirement_index(void) const = 0;
      virtual IndexSpaceID get_match_space(void) const = 0;
      virtual Operation* get_operation(void) const = 0;
      virtual const PhysicalTraceInfo& get_trace_info(void) const = 0;
      void pack_collective_analysis(Serializer &rez,
          AddressSpaceID target, std::set<RtEvent> &applied_events) const;
      virtual void add_analysis_reference(void) = 0;
      virtual bool remove_analysis_reference(void) = 0;
    };

    /**
     * \class RemoteCollectiveAnalysis
     * This class contains the needed data structures for representing
     * a physical analysis when registered on a remote collective view.
     */
    class RemoteCollectiveAnalysis : public CollectiveAnalysis,
                                     public Collectable {
    public:
      RemoteCollectiveAnalysis(size_t ctx_index, unsigned req_index,
                               IndexSpaceID match_space, RemoteOp *op,
                               Deserializer &derez, Runtime *runtime);
      virtual ~RemoteCollectiveAnalysis(void);
      virtual size_t get_context_index(void) const { return context_index; }
      virtual unsigned get_requirement_index(void) const
        { return requirement_index; }
      virtual IndexSpaceID get_match_space(void) const { return match_space; }
      virtual Operation* get_operation(void) const;
      virtual const PhysicalTraceInfo& get_trace_info(void) const
        { return trace_info; }
      virtual void add_analysis_reference(void) { add_reference(); }
      virtual bool remove_analysis_reference(void) 
        { return remove_reference(); }
      static RemoteCollectiveAnalysis* unpack(Deserializer &derez,
                Runtime *runtime, std::set<RtEvent> &ready_events);
    public:
      const size_t context_index;
      const unsigned requirement_index;
      const IndexSpaceID match_space;
      RemoteOp *const operation;
      const PhysicalTraceInfo trace_info;
    };

    /**
     * \class CollectiveCopyFillAnalysis
     * This is an intermediate base class for analyses that helps support
     * performing collective copies and fills on a destination collective
     * instance. It works be registering itself with the local collective
     * instance for its point so any collective copies and fills can be
     * attributed to the correct operation. After the analysis is done
     * then the analysis will unregister itself with the collecitve instance.
     */
    class CollectiveCopyFillAnalysis : public RegistrationAnalysis,
                                       public CollectiveAnalysis {
    public:
      CollectiveCopyFillAnalysis(Runtime *rt, Operation *op, unsigned index,
                                 RegionNode *node, bool on_heap,
                                 const PhysicalTraceInfo &trace_info,
                                 bool exclusive);
      CollectiveCopyFillAnalysis(Runtime *rt, 
                                 AddressSpaceID src, AddressSpaceID prev,
                                 Operation *op, unsigned index, 
                                 RegionNode *node, bool on_heap, 
                                 std::vector<PhysicalManager*> &&target_insts,
                                 LegionVector<
                                    FieldMaskSet<InstanceView> > &&target_views,
                                 std::vector<IndividualView*> &&source_views,
                                 const PhysicalTraceInfo &trace_info,
                                 CollectiveMapping *collective_mapping,
                                 bool first_local, bool exclusive);
      virtual ~CollectiveCopyFillAnalysis(void) { }
    public:
      virtual size_t get_context_index(void) const { return context_index; }
      virtual unsigned get_requirement_index(void) const { return index; }
      virtual IndexSpaceID get_match_space(void) const 
        { return get_collective_match_space(); }
      virtual Operation* get_operation(void) const { return op; }
      virtual const PhysicalTraceInfo& get_trace_info(void) const 
        { return trace_info; }
      virtual void add_analysis_reference(void) { add_reference(); }
      virtual bool remove_analysis_reference(void)
        { return remove_reference(); } 
      virtual RtEvent perform_traversal(RtEvent precondition,
                                        const VersionInfo &version_info,
                                        std::set<RtEvent> &applied_events);
    };

    /**
     * \class ValidInstAnalysis
     * For finding valid instances in equivalence set trees
     */
    class ValidInstAnalysis : public PhysicalAnalysis,
                              public LegionHeapify<ValidInstAnalysis> {
    public:
      ValidInstAnalysis(Runtime *rt, Operation *op, unsigned index,
                        IndexSpaceExpression *expr, ReductionOpID redop = 0);
      ValidInstAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                        Operation *op,unsigned index,IndexSpaceExpression *expr,
                        ValidInstAnalysis *target, ReductionOpID redop);
      ValidInstAnalysis(const ValidInstAnalysis &rhs) = delete;
      virtual ~ValidInstAnalysis(void);
    public:
      ValidInstAnalysis& operator=(const ValidInstAnalysis &rhs) = delete;
    public:
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
    public:
      static void handle_remote_request_instances(Deserializer &derez, 
                                     Runtime *rt, AddressSpaceID previous);
    public:
      const ReductionOpID redop;
      ValidInstAnalysis *const target_analysis;
    };

    /**
     * \class InvalidInstAnalysis
     * For finding which of a set of instances are not valid across
     * a set of equivalence sets
     */
    class InvalidInstAnalysis : public PhysicalAnalysis,
                                public LegionHeapify<InvalidInstAnalysis> {
    public:
      InvalidInstAnalysis(Runtime *rt, Operation *op, unsigned index,
                          IndexSpaceExpression *expr,
                          const FieldMaskSet<LogicalView> &valid_instances);
      InvalidInstAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                        Operation *op, unsigned index, 
                        IndexSpaceExpression *expr, InvalidInstAnalysis *target,
                        const FieldMaskSet<LogicalView> &valid_instances);
      InvalidInstAnalysis(const InvalidInstAnalysis &rhs) = delete;
      virtual ~InvalidInstAnalysis(void);
    public:
      InvalidInstAnalysis& operator=(const InvalidInstAnalysis &rhs) = delete;
    public:
      inline bool has_invalid(void) const
      { return ((recorded_instances != NULL) && !recorded_instances->empty()); }
    public:
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
    public:
      static void handle_remote_request_invalid(Deserializer &derez, 
                                     Runtime *rt, AddressSpaceID previous);
    public:
      const FieldMaskSet<LogicalView> valid_instances;
      InvalidInstAnalysis *const target_analysis;
    };

    /**
     * \class AntivalidInstAnalysis
     * For checking that some views are not in the set of valid instances
     */
    class AntivalidInstAnalysis : public PhysicalAnalysis,
                                  public LegionHeapify<AntivalidInstAnalysis> {
    public:
      AntivalidInstAnalysis(Runtime *rt, Operation *op, unsigned index,
                          IndexSpaceExpression *expr,
                          const FieldMaskSet<LogicalView> &anti_instances);
      AntivalidInstAnalysis(Runtime *rt, AddressSpaceID src,AddressSpaceID prev,
                      Operation *op, unsigned index, 
                      IndexSpaceExpression *expr, AntivalidInstAnalysis *target,
                      const FieldMaskSet<LogicalView> &anti_instances);
      AntivalidInstAnalysis(const AntivalidInstAnalysis &rhs) = delete;
      virtual ~AntivalidInstAnalysis(void);
    public:
      AntivalidInstAnalysis& operator=(const AntivalidInstAnalysis &r) = delete;
    public:
      inline bool has_antivalid(void) const
      { return ((recorded_instances != NULL) && !recorded_instances->empty()); }
    public:
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
    public:
      static void handle_remote_request_antivalid(Deserializer &derez, 
                                     Runtime *rt, AddressSpaceID previous);
    public:
      const FieldMaskSet<LogicalView> antivalid_instances;
      AntivalidInstAnalysis *const target_analysis;
    };

    /**
     * \class UpdateAnalysis
     * For performing updates on equivalence set trees
     */
    class UpdateAnalysis : public CollectiveCopyFillAnalysis,
                           public LegionHeapify<UpdateAnalysis> {
    public:
      UpdateAnalysis(Runtime *rt, Operation *op, unsigned index,
                     const RegionRequirement &req, RegionNode *node,
                     const PhysicalTraceInfo &trace_info,
                     const ApEvent precondition, const ApEvent term_event,
                     const bool check_initialized, const bool record_valid);
      UpdateAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                     Operation *op, unsigned index,
                     const RegionUsage &usage, RegionNode *node, 
                     std::vector<PhysicalManager*> &&target_instances,
                     LegionVector<FieldMaskSet<InstanceView> > &&target_views,
                     std::vector<IndividualView*> &&source_views,
                     const PhysicalTraceInfo &trace_info,
                     CollectiveMapping *collective_mapping,
                     const RtEvent user_registered,
                     const ApEvent precondition, const ApEvent term_event,
                     const bool check_initialized, const bool record_valid,
                     const bool first_local);
      UpdateAnalysis(const UpdateAnalysis &rhs) = delete;
      virtual ~UpdateAnalysis(void);
    public:
      UpdateAnalysis& operator=(const UpdateAnalysis &rhs) = delete;
    public:
      bool has_output_updates(void) const 
        { return (output_aggregator != NULL); }
    public:
      void record_uninitialized(const FieldMask &uninit,
                                std::set<RtEvent> &applied_events);
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition, 
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
      virtual RtEvent perform_registration(RtEvent precondition,
                                           const RegionUsage &usage,
                                           std::set<RtEvent> &registered_events,
                                           ApEvent init_precondition,
                                           ApEvent termination_event,
                                           ApEvent &instances_ready,
                                           bool symbolic = false);
      virtual ApEvent perform_output(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
    public:
      static void handle_remote_updates(Deserializer &derez, Runtime *rt,
                                        AddressSpaceID previous);
    public:
      const RegionUsage usage;
      const ApEvent precondition;
      const ApEvent term_event;
      const bool check_initialized;
      const bool record_valid;
    public:
      // Have to lock the analysis to access these safely
      std::map<RtEvent,CopyFillAggregator*> input_aggregators;
      CopyFillAggregator *output_aggregator;
      std::set<RtEvent> guard_events;
      // For tracking uninitialized data
      FieldMask uninitialized;
      RtUserEvent uninitialized_reported;
      // For remote tracking
      RtEvent remote_user_registered;
      RtUserEvent user_registered;
    public:
      // Event for when target instances are ready
      ApEvent instances_ready;
    };

    /**
     * \class AcquireAnalysis
     * For performing acquires on equivalence set trees
     */
    class AcquireAnalysis : public RegistrationAnalysis,
                            public LegionHeapify<AcquireAnalysis> {
    public: 
      AcquireAnalysis(Runtime *rt, Operation *op, unsigned index,
                      RegionNode *node, const PhysicalTraceInfo &t_info);
      AcquireAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                      Operation *op, unsigned index, RegionNode *node,
                      AcquireAnalysis *target,
                      const PhysicalTraceInfo &trace_info,
                      CollectiveMapping *collective_mapping,
                      bool first_local);
      AcquireAnalysis(const AcquireAnalysis &rhs) = delete;
      virtual ~AcquireAnalysis(void);
    public:
      AcquireAnalysis& operator=(const AcquireAnalysis &rhs) = delete;
    public:
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition, 
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
    public:
      static void handle_remote_acquires(Deserializer &derez, Runtime *rt,
                                         AddressSpaceID previous); 
    public:
      AcquireAnalysis *const target_analysis;
    };

    /**
     * \class ReleaseAnalysis
     * For performing releases on equivalence set trees
     */
    class ReleaseAnalysis : public CollectiveCopyFillAnalysis,
                            public LegionHeapify<ReleaseAnalysis> {
    public:
      ReleaseAnalysis(Runtime *rt, Operation *op, unsigned index,
                      ApEvent precondition, RegionNode *node,
                      const PhysicalTraceInfo &trace_info);
      ReleaseAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                      Operation *op, unsigned index, RegionNode *node,
                      ApEvent precondition, ReleaseAnalysis *target, 
                      std::vector<PhysicalManager*> &&target_instances,
                      LegionVector<FieldMaskSet<InstanceView> > &&target_views,
                      std::vector<IndividualView*> &&source_views,
                      const PhysicalTraceInfo &info,
                      CollectiveMapping *mapping, const bool first_local);
      ReleaseAnalysis(const ReleaseAnalysis &rhs) = delete;
      virtual ~ReleaseAnalysis(void);
    public:
      ReleaseAnalysis& operator=(const ReleaseAnalysis &rhs) = delete;
    public:
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
    public:
      static void handle_remote_releases(Deserializer &derez, Runtime *rt,
                                         AddressSpaceID previous);
    public:
      const ApEvent precondition;
      ReleaseAnalysis *const target_analysis;
    public:
      // Can only safely be accessed when analysis is locked
      CopyFillAggregator *release_aggregator;
    };

    /**
     * \class CopyAcrossAnalysis
     * For performing copy across traversals on equivalence set trees
     */
    class CopyAcrossAnalysis : public PhysicalAnalysis,
                               public LegionHeapify<CopyAcrossAnalysis> {
    public:
      CopyAcrossAnalysis(Runtime *rt, Operation *op, 
                         unsigned src_index, unsigned dst_index,
                         const RegionRequirement &src_req,
                         const RegionRequirement &dst_req,
                         const InstanceSet &target_instances,
                         const LegionVector<
                             FieldMaskSet<InstanceView> > &target_views,
                         const std::vector<IndividualView*> &source_views,
                         const ApEvent precondition, const ApEvent dst_ready,
                         const PredEvent pred_guard, const ReductionOpID redop,
                         const std::vector<unsigned> &src_indexes,
                         const std::vector<unsigned> &dst_indexes,
                         const PhysicalTraceInfo &trace_info,
                         const bool perfect);
      CopyAcrossAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                         Operation *op, unsigned src_index, unsigned dst_index,
                         const RegionUsage &src_usage, 
                         const RegionUsage &dst_usage,
                         const LogicalRegion src_region,
                         const LogicalRegion dst_region,
                         const ApEvent dst_ready,
                         std::vector<PhysicalManager*> &&target_instances,
                         LegionVector<
                              FieldMaskSet<InstanceView> > &&target_views,
                         std::vector<IndividualView*> &&source_views,
                         const ApEvent precondition,
                         const PredEvent pred_guard, const ReductionOpID redop,
                         const std::vector<unsigned> &src_indexes,
                         const std::vector<unsigned> &dst_indexes,
                         const PhysicalTraceInfo &trace_info,
                         const bool perfect);
      CopyAcrossAnalysis(const CopyAcrossAnalysis &rhs) = delete;
      virtual ~CopyAcrossAnalysis(void);
    public:
      CopyAcrossAnalysis& operator=(const CopyAcrossAnalysis &rhs) = delete;
    public:
      bool has_across_updates(void) const 
        { return (across_aggregator != NULL); }
      void record_uninitialized(const FieldMask &uninit,
                                std::set<RtEvent> &applied_events);
      CopyFillAggregator* get_across_aggregator(void);
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_updates(RtEvent precondition, 
                                      std::set<RtEvent> &applied_events,
                                      const bool already_deferred = false);
      virtual ApEvent perform_output(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
    public:
      static inline FieldMask initialize_mask(const std::vector<unsigned> &idxs)
      {
        FieldMask result;
        for (unsigned idx = 0; idx < idxs.size(); idx++)
          result.set_bit(idxs[idx]);
        return result;
      }
      static void handle_remote_copies_across(Deserializer &derez, Runtime *rt,
                                              AddressSpaceID previous); 
      static std::vector<CopyAcrossHelper*> create_across_helpers(
                            const FieldMask &src_mask,
                            const FieldMask &dst_mask,
                            const std::vector<PhysicalManager*> &dst_instances,
                            const std::vector<unsigned> &src_indexes,
                            const std::vector<unsigned> &dst_indexes);
      static std::vector<PhysicalManager*> convert_instances(
                                        const InstanceSet &instances);
    public:
      const FieldMask src_mask;
      const FieldMask dst_mask;
      const unsigned src_index;
      const unsigned dst_index;
      const RegionUsage src_usage;
      const RegionUsage dst_usage;
      const LogicalRegion src_region;
      const LogicalRegion dst_region;
      const ApEvent targets_ready;
      const std::vector<PhysicalManager*> target_instances;
      const LegionVector<FieldMaskSet<InstanceView> > target_views;
      const std::vector<IndividualView*> source_views;
      const ApEvent precondition;
      const PredEvent pred_guard;
      const ReductionOpID redop;
      const std::vector<unsigned> src_indexes;
      const std::vector<unsigned> dst_indexes;
      const std::vector<CopyAcrossHelper*> across_helpers;
      const PhysicalTraceInfo trace_info;
      const bool perfect;
    public:
      // Can only safely be accessed when analysis is locked
      FieldMask uninitialized;
      RtUserEvent uninitialized_reported;
      FieldMaskSet<IndexSpaceExpression> local_exprs;
      std::vector<ApEvent> copy_events;
      std::set<RtEvent> guard_events;
    protected:
      CopyFillAggregator *across_aggregator;
      RtUserEvent aggregator_guard; // Guard event for the aggregator
    };

    /**
     * \class OverwriteAnalysis 
     * For performing overwrite traversals on equivalence set trees
     */
    class OverwriteAnalysis : public PhysicalAnalysis,
                              public LegionHeapify<OverwriteAnalysis> {
    public:
      OverwriteAnalysis(Runtime *rt, Operation *op, unsigned index,
                        const RegionUsage &usage, IndexSpaceExpression *expr,
                        LogicalView *view, const FieldMask &mask,
                        const PhysicalTraceInfo &trace_info,
                        CollectiveMapping *collective_mapping,
                        const ApEvent precondition,
                        const PredEvent true_guard = PredEvent::NO_PRED_EVENT,
                        const PredEvent false_guard = PredEvent::NO_PRED_EVENT,
                        const bool add_restriction = false,
                        const bool first_local = true);
      // Also local but with a full set of instances 
      OverwriteAnalysis(Runtime *rt, Operation *op, unsigned index,
                        const RegionUsage &usage, IndexSpaceExpression *expr,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const bool add_restriction = false);
      // Also local but with a full set of views
      OverwriteAnalysis(Runtime *rt, Operation *op, unsigned index,
                        const RegionUsage &usage, IndexSpaceExpression *expr,
                        const FieldMaskSet<LogicalView> &overwrite_views,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const bool add_restriction = false);
      OverwriteAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                        Operation *op, unsigned index,
                        IndexSpaceExpression *expr, const RegionUsage &usage, 
                        FieldMaskSet<LogicalView> &views,
                        FieldMaskSet<InstanceView> &reduction_views,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const PredEvent true_guard,
                        const PredEvent false_guard,
                        CollectiveMapping *mapping,
                        const bool first_local,
                        const bool add_restriction);
      OverwriteAnalysis(const OverwriteAnalysis &rhs) = delete;
      virtual ~OverwriteAnalysis(void);
    public:
      OverwriteAnalysis& operator=(const OverwriteAnalysis &rhs) = delete;
    public:
      bool has_output_updates(void) const 
        { return (output_aggregator != NULL); }
    public:
      virtual RtEvent perform_traversal(RtEvent precondition,
                                        const VersionInfo &version_info,
                                        std::set<RtEvent> &applied_events);
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition, 
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
      virtual RtEvent perform_registration(RtEvent precondition,
                                           const RegionUsage &usage,
                                           std::set<RtEvent> &applied_events,
                                           ApEvent init_precondition,
                                           ApEvent termination_event,
                                           ApEvent &instances_ready,
                                           bool symbolic = false);
      virtual ApEvent perform_output(RtEvent precondition,
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
    public:
      RtEvent convert_views(LogicalRegion region, const InstanceSet &targets,
                            unsigned analysis_index = 0);
      static void handle_remote_overwrites(Deserializer &derez, Runtime *rt,
                                           AddressSpaceID previous); 
    public:
      const RegionUsage usage;
      const PhysicalTraceInfo trace_info;
      FieldMaskSet<LogicalView> views;
      FieldMaskSet<InstanceView> reduction_views;
      const ApEvent precondition;
      const PredEvent true_guard;
      const PredEvent false_guard;
      const bool add_restriction;
    public:
      // Can only safely be accessed when analysis is locked
      CopyFillAggregator *output_aggregator;
    protected:
      std::vector<PhysicalManager*> target_instances;
      LegionVector<FieldMaskSet<InstanceView> > target_views;
      std::map<InstanceView*,size_t> collective_arrivals;
    };

    /**
     * \class FilterAnalysis
     * For performing filter traversals on equivalence set trees
     */
    class FilterAnalysis : public RegistrationAnalysis,
                           public LegionHeapify<FilterAnalysis> {
    public:
      FilterAnalysis(Runtime *rt, Operation *op, unsigned index,
                     RegionNode *node, const PhysicalTraceInfo &trace_info,
                     const bool remove_restriction = false);
      FilterAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                     Operation *op, unsigned index, RegionNode *node,
                     const PhysicalTraceInfo &trace_info,
                     const FieldMaskSet<InstanceView> &filter_views,
                     CollectiveMapping *mapping, const bool first_local,
                     const bool remove_restriction);
      FilterAnalysis(const FilterAnalysis &rhs) = delete;
      virtual ~FilterAnalysis(void);
    public:
      FilterAnalysis& operator=(const FilterAnalysis &rhs) = delete;
    public:
      virtual RtEvent perform_traversal(RtEvent precondition,
                                        const VersionInfo &version_info,
                                        std::set<RtEvent> &applied_events);
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition, 
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
    public:
      static void handle_remote_filters(Deserializer &derez, Runtime *rt,
                                        AddressSpaceID previous);
    public:
      FieldMaskSet<InstanceView> filter_views;
      const bool remove_restriction;
    };

    /**
     * \class CloneAnalysis
     * For cloning into an equivalence set from a set of other equivalence
     * sets. This usually is done by VirtualCloseOps.
     */
    class CloneAnalysis : public PhysicalAnalysis,
                          public LegionHeapify<CloneAnalysis> {
    public:
      CloneAnalysis(Runtime *rt, IndexSpaceExpression *expr, Operation *op,
                    unsigned index, FieldMaskSet<EquivalenceSet> &&sources);
      CloneAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                    IndexSpaceExpression *expr, Operation *op,
                    unsigned index, FieldMaskSet<EquivalenceSet> &&sources);
      CloneAnalysis(const CloneAnalysis &rhs) = delete;
      virtual ~CloneAnalysis(void);
    public:
      CloneAnalysis& operator=(const CloneAnalysis &rhs) = delete;
    public:
      virtual bool perform_analysis(EquivalenceSet *set,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred = false);
      virtual RtEvent perform_remote(RtEvent precondition, 
                                     std::set<RtEvent> &applied_events,
                                     const bool already_deferred = false);
    public:
      static void handle_remote_clones(Deserializer &derez, Runtime *rt,
                                       AddressSpaceID previous);
    public:
      const FieldMaskSet<EquivalenceSet> sources;
    };

    /**
     * \class CollectiveRefinementTree
     * This class provides a base class for performing analyses inside of
     * an equivalence set that might have to deal with aliased names for
     * physical instances due to collective views. It does this by 
     * constructing a tree of set of instances that are all performing
     * the same analysis. As new views are traversed, the analysis can
     * fracture into subsets of instances that all have the same state.
     */
    template<typename T>
    class CollectiveRefinementTree {
    public:
      CollectiveRefinementTree(CollectiveView *view);
      CollectiveRefinementTree(std::vector<DistributedID> &&inst_dids);
      ~CollectiveRefinementTree(void);
    public:
      const FieldMask& get_refinement_mask(void) const 
        { return refinements.get_valid_mask(); }
      const std::vector<DistributedID>& get_instances(void) const;
      // Traverse the tree and perform refinements as necessary 
      template<typename... Args>
      void traverse(InstanceView *view, const FieldMask &mask, Args&&... args);
      // Visit just the leaves of the analysis
      template<typename... Args>
      void visit_leaves(const FieldMask &mask, Args&&... args)
      {
        for (typename FieldMaskSet<T>::const_iterator it = 
              refinements.begin(); it != refinements.end(); it++)
        {
          const FieldMask &overlap = mask & it->second;
          if (!overlap)
            continue;
          it->first->visit_leaves(overlap, args...);
        }
        const FieldMask local_mask = mask - refinements.get_valid_mask();
        if (!!local_mask)
          static_cast<T*>(this)->visit_leaf(local_mask, args...);
      }
      InstanceView* get_instance_view(InnerContext *context, 
                                      RegionTreeID tid) const;
    public:
      // Helper methods for traversing the total and partial valid views
      // These should only be called on the root with collective != NULL
      void traverse_total(const FieldMask &mask, IndexSpaceExpression *set_expr,
                          const FieldMaskSet<LogicalView> &total_valid_views);
      void traverse_partial(const FieldMask &mask, const LegionMap<LogicalView*,
                     FieldMaskSet<IndexSpaceExpression> > &partial_valid_views);
    protected:
      CollectiveView *const collective;
      const std::vector<DistributedID> inst_dids;
    private:
      FieldMaskSet<T> refinements;
    };

    /**
     * \class MakeCollectiveValid
     * This class helps in the aliasing analysis when determining how
     * to issue copies to update collective views. It builds an instance
     * refinement tree to find the valid expressions for each group of
     * instances and then can use that to compute the updates
     */
    class MakeCollectiveValid : 
      public CollectiveRefinementTree<MakeCollectiveValid>,
      public LegionHeapify<MakeCollectiveValid> {
    public:
      MakeCollectiveValid(CollectiveView *view,
          const FieldMaskSet<IndexSpaceExpression> &needed_exprs);
      MakeCollectiveValid(std::vector<DistributedID> &&insts,
          const FieldMaskSet<IndexSpaceExpression> &needed_exprs,
          const FieldMaskSet<IndexSpaceExpression> &valid_expr,
          const FieldMask &mask, InstanceView *inst_view);
    public:
      MakeCollectiveValid* clone(InstanceView *view, const FieldMask &mask,
                                 std::vector<DistributedID> &&insts) const;
      void analyze(InstanceView *view, const FieldMask &mask,
                   IndexSpaceExpression *expr);
      void analyze(InstanceView *view, const FieldMask &mask,
                   const FieldMaskSet<IndexSpaceExpression> &exprs);
      void visit_leaf(const FieldMask &mask, InnerContext *context,
         RegionTreeForest *forest, RegionTreeID tid,
         LegionMap<InstanceView*,FieldMaskSet<IndexSpaceExpression> > &updates);
    public:
      InstanceView *const view;
    protected:
      // Expression fields that we need to make valid for all instances
      const FieldMaskSet<IndexSpaceExpression> &needed_exprs;
      // Expression fields that are valid for these instances
      FieldMaskSet<IndexSpaceExpression> valid_exprs;
    };

    /**
     * \class CollectiveAntiAlias 
     * This class helps with the alias analysis when performing a check
     * to see which collective instances are not considered valid
     * It is also used by TraceViewSet::dominates to test for dominance
     * of collective views which is the same as testing that the
     * collective view 
     */
    class CollectiveAntiAlias : 
      public CollectiveRefinementTree<CollectiveAntiAlias>,
      public LegionHeapify<CollectiveAntiAlias> {
    public:
      CollectiveAntiAlias(CollectiveView *view);
      CollectiveAntiAlias(std::vector<DistributedID> &&insts,
          const FieldMaskSet<IndexSpaceExpression> &valid_expr,
          const FieldMask &mask);
    public:
      CollectiveAntiAlias* clone(InstanceView *view, const FieldMask &mask,
                                   std::vector<DistributedID> &&insts) const;
      void analyze(InstanceView *view, const FieldMask &mask,
                   IndexSpaceExpression *expr);
      void analyze(InstanceView *view, const FieldMask &mask,
                   const FieldMaskSet<IndexSpaceExpression> &exprs);
      void visit_leaf(const FieldMask &mask, FieldMask &allvalid_mask,
          IndexSpaceExpression *expr, RegionTreeForest *forest,
          const FieldMaskSet<IndexSpaceExpression> &partial_valid_exprs);
      // This version used by TraceViewSet::dominates to find
      // non-dominating overlaps
      void visit_leaf(const FieldMask &mask, FieldMask &allvalid_mask,
          FieldMaskSet<IndexSpaceExpression> &non_dominated,
          IndexSpaceExpression *expr, RegionTreeForest *forest);
      // This version used by TraceViewSet::antialias_collective_view
      // to get the names of the new views to use for instances
      void visit_leaf(const FieldMask &mask, FieldMask &allvalid_mask,
          TraceViewSet &view_set, FieldMaskSet<InstanceView> &alt_views);
    protected:
      // Expression fields that are valid for these instances
      FieldMaskSet<IndexSpaceExpression> valid_exprs;
    };
    
    /**
     * \class InitializeCollectiveReduction
     * This class helps with aliasing analysis for recording collective
     * reduction views and figuring out where they are already valid
     */
    class InitializeCollectiveReduction :
      public CollectiveRefinementTree<InitializeCollectiveReduction>,
      public LegionHeapify<InitializeCollectiveReduction> {
    public:
      InitializeCollectiveReduction(AllreduceView *view,
                                    RegionTreeForest *forest,
                                    IndexSpaceExpression *expr);
      InitializeCollectiveReduction(std::vector<DistributedID> &&insts,
                                    RegionTreeForest *forest,
                                    IndexSpaceExpression *expr,
                                    InstanceView *view,
                          const FieldMaskSet<IndexSpaceExpression> &remainders,
                                    const FieldMask &covered);
    public:
      InitializeCollectiveReduction* clone(InstanceView *view,
          const FieldMask &mask, std::vector<DistributedID> &&insts) const;
      void analyze(InstanceView *view, const FieldMask &mask,
                   IndexSpaceExpression *expr);
      void analyze(InstanceView *view, const FieldMask &mask,
                   const FieldMaskSet<IndexSpaceExpression> &exprs);
      // Check for ABA problem
      void visit_leaf(const FieldMask &mask, IndexSpaceExpression *expr,
                      bool &failure);
      // Report out any fill operations that we need to perform
      void visit_leaf(const FieldMask &mask, InnerContext *context,
         UpdateAnalysis &analysis, CopyFillAggregator *&fill_aggregator,
         FillView *fill_view, RegionTreeID tid, EquivalenceSet *eq_set,
         DistributedID eq_did, std::map<unsigned,std::list<std::pair<
          InstanceView*,IndexSpaceExpression*> > > &reduction_instances);
    public:
      RegionTreeForest *const forest;
      IndexSpaceExpression *const needed_expr;
      InstanceView *const view;
    protected:
      // Expressions for which we still need fill values
      FieldMaskSet<IndexSpaceExpression> remainder_exprs;
      FieldMask found_covered;
    };

    template<typename T>
    struct SubscriberInvalidations : 
      public LegionHeapify<SubscriberInvalidations<T> > {
      SubscriberInvalidations(void) : all_subscribers_finished(false) { }
      FieldMaskSet<T> subscribers;
      std::vector<T*> finished;
      bool all_subscribers_finished;
    };

    /**
     * \class EqSetTracker
     * This is an abstract class that provides an interface for
     * recording the equivalence sets that result from ray tracing
     * an equivalence set tree for a given index space expression.
     */
    class EqSetTracker {
    public:
      EqSetTracker(LocalLock &lock);
      virtual ~EqSetTracker(void) { }
    public:
      void record_equivalence_sets(
          InnerContext *context, const FieldMask &mask,
          FieldMaskSet<EquivalenceSet> &eq_sets,
          FieldMaskSet<EqKDTree> &to_create,
          std::map<EqKDTree*,Domain> &creation_rects,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
          FieldMaskSet<EqKDTree> &subscriptions,
          AddressSpaceID source, unsigned expected_responses,
          std::vector<RtEvent> &ready_events);
    public:
      virtual void add_subscription_reference(unsigned count = 1) = 0;
      virtual bool remove_subscription_reference(unsigned count = 1) = 0;
    public:
      virtual RegionTreeID get_region_tree_id(void) const = 0;
      virtual IndexSpaceExpression* get_tracker_expression(void) const = 0;
      virtual size_t count_outstanding_requests(void) const = 0;
      virtual ReferenceSource get_reference_source_kind(void) const = 0;
    public:
      bool invalidate_equivalence_sets(Runtime *runtime, const FieldMask &mask,
                    EqKDTree *tree, AddressSpaceID source, 
                    std::vector<RtEvent> &invalidated_events);
      typedef SubscriberInvalidations<EqKDTree> TreeInvalidations;
      void cancel_subscriptions(Runtime *runtime,
          const LegionMap<AddressSpaceID,TreeInvalidations> &to_cancel,
          std::vector<RtEvent> *cancelled_events = NULL);
#if 0
      bool finish_subscription(EqKDTree *owner, AddressSpaceID space,
                               const FieldMask &mask);
#endif
      typedef SubscriberInvalidations<EqSetTracker> TrackerInvalidations;
      static void invalidate_subscriptions(Runtime *runtime, EqKDTree *source,
          LegionMap<AddressSpaceID,TrackerInvalidations> &subscribers,
          std::vector<RtEvent> &applied_events);
      static void handle_cancel_subscription(Deserializer &derez,
          Runtime *runtime, AddressSpaceID source);
      static void handle_invalidate_subscription(Deserializer &derez,
          Runtime *runtime, AddressSpaceID source);
      static void handle_equivalence_set_creation(Deserializer &derez,
                                                  Runtime *runtime);
      static void handle_equivalence_set_reuse(Deserializer &derez,
                                               Runtime *runtime);
      void record_pending_equivalence_set(EquivalenceSet *set, 
                                          const FieldMask &mask);
      static void handle_pending_equivalence_set(Deserializer &derez,
                                                 Runtime *runtime);
    protected:
      void record_subscriptions(AddressSpaceID source,
                const FieldMaskSet<EqKDTree> &new_subs);
      void record_creation_sets(FieldMaskSet<EqKDTree> &to_create,
         std::map<EqKDTree*,Domain> &creation_rects, AddressSpaceID source,
         std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs);
      void extract_creation_sets(const FieldMask &mask,
         LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > &create_now,
         LegionMap<Domain,FieldMask> &create_now_rectangles,
         std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs);
      void create_new_equivalence_sets(InnerContext *context,
         const FieldMask &mask, std::vector<RtEvent> &ready_events,
         LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > &create_now,
         LegionMap<Domain,FieldMask> &create_now_rectangles,
         std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs);
      struct SourceState : public FieldSet<Domain> {
      public:
        SourceState(void) : source_expr(NULL), source_volume(0) { }
        SourceState(const FieldMask &m) : 
          FieldSet(m), source_expr(NULL), source_volume(0) { }
        ~SourceState(void);
      public:
        IndexSpaceExpression* get_expression(void) const;
        void set_expression(IndexSpaceExpression *expr);
      public:
        IndexSpaceExpression *source_expr;
        size_t source_volume;
      };
      bool check_for_congruent_source_equivalence_sets(
          FieldSet<Domain> &dest, Runtime *runtime,
          std::vector<RtEvent> &ready_events,
          FieldMaskSet<EquivalenceSet> &pending_sets,
          FieldMaskSet<EquivalenceSet> &unique_sources,
          LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > &create_now,
          std::map<EquivalenceSet*,LegionList<SourceState> > &creation_sources);
      void extract_remote_notifications(const FieldMask &mask,
          AddressSpaceID local_space,
          LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > &create_now,
          LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > &to_notify);
      RtEvent initialize_new_equivalence_set(EquivalenceSet *set,
          const FieldMask &mask, Runtime *runtime,
          std::map<EquivalenceSet*,LegionList<SourceState> > &creation_sources);
    protected:
      LocalLock &tracker_lock;
      FieldMaskSet<EquivalenceSet> equivalence_sets;
      FieldMaskSet<EquivalenceSet> pending_equivalence_sets;
      // These are the EqKDTree objects that we are currently subscribed to
      // for different fields in each address space, this data mirrors the 
      // same data structure in EqKDNode
      LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > current_subscriptions;
      // These all help with the creation of equivalence sets for which we
      // are the first request to access them 
      LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> >     creation_requests;
      LegionMap<Domain,FieldMask>                         creation_rectangles;
      std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > creation_sources;
      LegionMap<unsigned,FieldMask>                       remaining_responses;
    };

    /**
     * \class EquivalenceSet
     * The equivalence set class tracks the physical state of a
     * set of points in a logical region for all the fields. There
     * is an owner node for the equivlance set that uses a ESI
     * protocol in order to manage local and remote copies of 
     * the equivalence set for each of the different fields.
     * It's also possible for the equivalence set to be refined
     * into sub equivalence sets which then subsum it's responsibility.
     */
    class EquivalenceSet : public DistributedCollectable,
                           public LegionHeapify<EquivalenceSet> {
    public:
      struct ReplicatedOwnerState : public LegionHeapify<ReplicatedOwnerState> {
      public:
        ReplicatedOwnerState(bool valid);
      public:
        inline bool is_valid(void) const { return valid; }
        inline bool is_subscribed(void) const { return subscribed; }
      public:
        std::vector<AddressSpaceID> children;
        RtUserEvent ready;
        bool valid;
        // We might not be subscribed to the logical owner space if this
        // replicated owner state was created as part of mapping
        bool subscribed;
      };
    public:
      struct DeferMakeOwnerArgs : public LgTaskArgs<DeferMakeOwnerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAKE_OWNER_TASK_ID;
      public:
        DeferMakeOwnerArgs(EquivalenceSet *s)
          : LgTaskArgs<DeferMakeOwnerArgs>(implicit_provenance), 
            set(s) { }
      public:
        EquivalenceSet *const set;
      };
      struct DeferApplyStateArgs : public LgTaskArgs<DeferApplyStateArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_APPLY_STATE_TASK_ID;
        typedef LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<LogicalView> > ExprLogicalViews; 
        typedef std::map<unsigned,std::list<std::pair<InstanceView*,
          IndexSpaceExpression*> > > ExprReductionViews;
        typedef LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<InstanceView> > ExprInstanceViews;
      public:
        DeferApplyStateArgs(EquivalenceSet *set,
                            const bool foward_to_owner,
                            std::vector<RtEvent> &applied_events,
                            ExprLogicalViews &valid_updates,
                            FieldMaskSet<IndexSpaceExpression> &init_updates,
                            ExprReductionViews &reduction_updates,
                            ExprInstanceViews &restricted_updates,
                            ExprInstanceViews &released_updates,
                            FieldMaskSet<CopyFillGuard> &read_only_updates,
                            FieldMaskSet<CopyFillGuard> &reduction_fill_updates,
                            TraceViewSet *precondition_updates,
                            TraceViewSet *anticondition_updates,
                            TraceViewSet *postcondition_updates);
        void release_references(void) const;
      public:
        EquivalenceSet *const set;
        ExprLogicalViews *const valid_updates;
        FieldMaskSet<IndexSpaceExpression> *const initialized_updates;
        ExprReductionViews *const reduction_updates;
        ExprInstanceViews *const restricted_updates;
        ExprInstanceViews *const released_updates;
        FieldMaskSet<CopyFillGuard> *const read_only_updates;
        FieldMaskSet<CopyFillGuard> *const reduction_fill_updates;
        TraceViewSet *precondition_updates;
        TraceViewSet *anticondition_updates;
        TraceViewSet *postcondition_updates;
        const RtUserEvent done_event;
        const bool forward_to_owner;
      };
    public:
      EquivalenceSet(Runtime *rt, DistributedID did,
                     AddressSpaceID logical_owner,
                     IndexSpaceExpression *expr,
                     RegionTreeID tid,
                     InnerContext *context,
                     bool register_now, 
                     CollectiveMapping *mapping = NULL);
      EquivalenceSet(const EquivalenceSet &rhs) = delete;
      virtual ~EquivalenceSet(void);
    public:
      EquivalenceSet& operator=(const EquivalenceSet &rhs) = delete;
      // Must be called while holding the lock
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner_space); }
    public:
      // From distributed collectable
      virtual void notify_invalid(void) { assert(false); }
      virtual void notify_local(void);
    public:
      // Analysis methods
      void initialize_set(const RegionUsage &usage,
                          const FieldMask &user_mask,
                          const bool restricted,
                          const InstanceSet &sources,
            const std::vector<IndividualView*> &corresponding);
      void analyze(PhysicalAnalysis &analysis,
                   IndexSpaceExpression *expr,
                   const bool expr_covers, 
                   FieldMask traversal_mask,
                   std::set<RtEvent> &deferral_events,
                   std::set<RtEvent> &applied_events,
                   const bool already_deferred);
      void find_valid_instances(ValidInstAnalysis &analysis,
                                IndexSpaceExpression *expr,
                                const bool expr_covers, 
                                const FieldMask &user_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool already_deferred = false);
      void find_invalid_instances(InvalidInstAnalysis &analysis,
                                IndexSpaceExpression *expr,
                                const bool expr_covers, 
                                const FieldMask &user_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool already_deferred = false);
      void find_antivalid_instances(AntivalidInstAnalysis &analysis,
                                IndexSpaceExpression *expr,
                                const bool expr_covers, 
                                const FieldMask &user_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool already_deferred = false);
      void update_set(UpdateAnalysis &analysis, IndexSpaceExpression *expr,
                      const bool expr_covers, FieldMask user_mask,
                      std::set<RtEvent> &deferral_events,
                      std::set<RtEvent> &applied_events,
                      const bool already_deferred = false);
      void acquire_restrictions(AcquireAnalysis &analysis, 
                                IndexSpaceExpression *expr,
                                const bool expr_covers, 
                                const FieldMask &acquire_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool already_deferred = false);
      void release_restrictions(ReleaseAnalysis &analysis,
                                IndexSpaceExpression *expr,
                                const bool expr_covers, 
                                const FieldMask &release_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool already_deferred = false);
      void issue_across_copies(CopyAcrossAnalysis &analysis,
                               const FieldMask &src_mask, 
                               IndexSpaceExpression *expr,
                               const bool expr_covers,
                               std::set<RtEvent> &deferral_events,
                               std::set<RtEvent> &applied_events,
                               const bool already_deferred = false);
      void overwrite_set(OverwriteAnalysis &analysis, 
                         IndexSpaceExpression *expr, const bool expr_covers,
                         const FieldMask &overwrite_mask,
                         std::set<RtEvent> &deferral_events,
                         std::set<RtEvent> &applied_events,
                         const bool already_deferred = false);
      void filter_set(FilterAnalysis &analysis, 
                      IndexSpaceExpression *expr, const bool expr_covers,
                      const FieldMask &filter_mask, 
                      std::set<RtEvent> &deferral_events,
                      std::set<RtEvent> &applied_events,
                      const bool already_deferred = false);
      void clone_set(CloneAnalysis &analysis,
                     IndexSpaceExpression *expr, const bool expr_covers,
                     const FieldMask &clone_mask,
                     std::set<RtEvent> &deferral_events,
                     std::vector<RtEvent> &applied_events,
                     const bool already_deferred = false);
    public:
      void initialize_collective_references(unsigned local_valid_refs);
      void remove_read_only_guard(CopyFillGuard *guard);
      void remove_reduction_fill_guard(CopyFillGuard *guard);
      void clone_from(const AddressSpaceID target_space, EquivalenceSet *src,
                      const FieldMask &clone_mask,
                      IndexSpaceExpression *clone_expr,
                      const bool forward_to_owner,
                      std::vector<RtEvent> &applied_events, 
                      const bool invalidate_overlap = false);
      RtEvent make_owner(AddressSpaceID owner, 
                         RtEvent precondition = RtEvent::NO_RT_EVENT);
      void update_tracing_valid_views(LogicalView *view,
                                      IndexSpaceExpression *expr,
                                      const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      const bool invalidates);
      void update_tracing_anti_views(LogicalView *view,
                                     IndexSpaceExpression *expr,
                                     const FieldMask &user_mask);
      RtEvent capture_trace_conditions(TraceConditionSet *target,
                                       AddressSpaceID target_space,
                                       IndexSpaceExpression *expr,
                                       const FieldMask &mask,
                                       RtUserEvent ready_event);
    protected:
      void defer_analysis(AutoTryLock &eq, PhysicalAnalysis &analysis,
                          const FieldMask &mask,
                          std::set<RtEvent> &deferral_events,
                          std::set<RtEvent> &applied_events,
                          const bool already_deferred);
      inline RtEvent chain_deferral_events(RtUserEvent deferral_event)
      {
        RtEvent continuation_pre;
        continuation_pre.id =
          next_deferral_precondition.exchange(deferral_event.id);
        return continuation_pre;
      }
      bool is_remote_analysis(PhysicalAnalysis &analysis,
                              FieldMask &traversal_mask, 
                              std::set<RtEvent> &deferral_events,
                              std::set<RtEvent> &applied_events,
                              const bool expr_covers);
    protected:
      template<typename T>
      void check_for_uninitialized_data(T &analysis, IndexSpaceExpression *expr,
                                  const bool expr_cover, FieldMask uninit,
                                  std::set<RtEvent> &applied_events) const;
      void update_initialized_data(IndexSpaceExpression *expr, 
                                   const bool expr_covers,
                                   const FieldMask &user_mask);
      template<typename T>
      void record_instances(IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &record_mask, 
                            const FieldMaskSet<T> &new_views);
      template<typename T>
      void record_unrestricted_instances(IndexSpaceExpression *expr,
                            const bool expr_covers, FieldMask record_mask, 
                            const FieldMaskSet<T> &new_views);
      bool record_partial_valid_instance(LogicalView *instance,
                                         IndexSpaceExpression *expr,
                                         FieldMask valid_mask,
                                         bool check_total_valid = true);
      void filter_valid_instance(InstanceView *to_filter,
                                 IndexSpaceExpression *expr,
                                 const bool expr_covers,
                                 const FieldMask &filter_mask);
      void filter_valid_instances(IndexSpaceExpression *expr, 
                                  const bool expr_covers, 
                                  const FieldMask &filter_mask,
           std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
           std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      void filter_unrestricted_instances(IndexSpaceExpression *expr,
                                         const bool expr_covers, 
                                         FieldMask filter_mask);
      void filter_reduction_instances(IndexSpaceExpression *expr,
           const bool covers, const FieldMask &mask,
           std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
           std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      void update_set_internal(CopyFillAggregator *&input_aggregator,
                               CopyFillGuard *previous_guard,
                               PhysicalAnalysis *analysis,
                               const RegionUsage &usage,
                               IndexSpaceExpression *expr, 
                               const bool expr_covers,
                               const FieldMask &user_mask,
                               const std::vector<PhysicalManager*> &targets,
                               const LegionVector<
                                   FieldMaskSet<InstanceView> > &target_views,
                               const std::vector<IndividualView*> &source_views,
                               const PhysicalTraceInfo &trace_info,
                               const bool record_valid);
      void make_instances_valid(CopyFillAggregator *&aggregator,
                               CopyFillGuard *previous_guard,
                               PhysicalAnalysis *analysis,
                               const bool track_events,
                               IndexSpaceExpression *expr,
                               const bool expr_covers,
                               const FieldMask &update_mask,
                               const std::vector<PhysicalManager*> &targets,
                               const LegionVector<
                                   FieldMaskSet<InstanceView> > &target_views,
                               const std::vector<IndividualView*> &source_views,
                               const PhysicalTraceInfo &trace_info,
                               const bool skip_check = false,
                               const ReductionOpID redop = 0,
                               CopyAcrossHelper *across_helper = NULL);
      void issue_update_copies_and_fills(InstanceView *target,
                                         PhysicalManager *target_manager,
                               const std::vector<IndividualView*> &source_views,
                                         CopyFillAggregator *&aggregator,
                                         CopyFillGuard *previous_guard,
                                         PhysicalAnalysis *analysis, 
                                         const bool track_events,
                                         IndexSpaceExpression *expr,
                                         const bool expr_covers,
                                         FieldMask update_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         const ReductionOpID redop,
                                         CopyAcrossHelper *across_helper);
      void record_reductions(UpdateAnalysis &analysis,
                             IndexSpaceExpression *expr,
                             const bool expr_covers,
                             const FieldMask &user_mask);
      void apply_reductions(const std::vector<PhysicalManager*> &targets,
                            const LegionVector<
                                    FieldMaskSet<InstanceView> > &target_views,
                            IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &reduction_mask,
                            CopyFillAggregator *&aggregator,
                            CopyFillGuard *previous_guard,
                            PhysicalAnalysis *analysis,
                            const bool track_events,
                            const PhysicalTraceInfo &trace_info,
                            FieldMaskSet<IndexSpaceExpression> *applied_exprs,
                            CopyAcrossHelper *across_helper = NULL);
      void apply_restricted_reductions(
                            const FieldMaskSet<InstanceView> &reduction_targets,
                            IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &reduction_mask,
                            CopyFillAggregator *&aggregator,
                            CopyFillGuard *previous_guard,
                            PhysicalAnalysis *analysis,
                            const bool track_events,
                            const PhysicalTraceInfo &trace_info,
                            FieldMaskSet<IndexSpaceExpression> *applied_exprs);
      void apply_reduction(InstanceView *target,PhysicalManager *target_manager,
                            IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &reduction_mask,
                            CopyFillAggregator *&aggregator,
                            CopyFillGuard *previous_guard,
                            PhysicalAnalysis *analysis,
                            const bool track_events,
                            const PhysicalTraceInfo &trace_info,
                            FieldMaskSet<IndexSpaceExpression> *applied_exprs,
                            CopyAcrossHelper *across_helper);
      void copy_out(IndexSpaceExpression *expr, const bool expr_covers,
                    const FieldMask &restricted_mask, 
                    const std::vector<PhysicalManager*> &target_instances,
                    const LegionVector<
                               FieldMaskSet<InstanceView> > &target_views,
                    PhysicalAnalysis *analysis,
                    const PhysicalTraceInfo &trace_info,
                    CopyFillAggregator *&aggregator);
      template<typename T>
      void copy_out(IndexSpaceExpression *expr, const bool expr_covers,
                    const FieldMask &restricted_mask,
                    const FieldMaskSet<T> &src_views,
                    PhysicalAnalysis *analysis,
                    const PhysicalTraceInfo &trace_info,
                    CopyFillAggregator *&aggregator);
      void predicate_fill_all(FillView *fill, const FieldMask &fill_mask,
                              IndexSpaceExpression *expr,
                              const bool expr_covers,
                              const PredEvent true_guard,
                              const PredEvent false_guard,
                              PhysicalAnalysis *analysis,
                              const PhysicalTraceInfo &trace_info,
                              CopyFillAggregator *&aggregator);
      void record_restriction(IndexSpaceExpression *expr, 
                              const bool expr_covers,
                              const FieldMask &restrict_mask,
                              InstanceView *restricted_view);
      void update_reductions(const unsigned fidx,
          std::list<std::pair<InstanceView*,IndexSpaceExpression*> > &updates);
      void update_released(IndexSpaceExpression *expr, const bool expr_covers,
                FieldMaskSet<InstanceView> &updates);
      void filter_initialized_data(IndexSpaceExpression *expr, 
          const bool expr_covers, const FieldMask &filter_mask, 
          std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL);
      void filter_restricted_instances(IndexSpaceExpression *expr, 
          const bool covers, const FieldMask &mask,
          std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
          std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      void filter_released_instances(IndexSpaceExpression *expr, 
          const bool covers, const FieldMask &mask,
          std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
          std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      bool find_fully_valid_fields(InstanceView *inst, FieldMask &inst_mask,
          IndexSpaceExpression *expr, const bool expr_covers) const;
      bool find_partial_valid_fields(InstanceView *inst, FieldMask &inst_mask,
          IndexSpaceExpression *expr, const bool expr_covers,
          FieldMaskSet<IndexSpaceExpression> &partial_valid_exprs) const;
    protected:
      void send_equivalence_set(AddressSpaceID target);
      void check_for_migration(PhysicalAnalysis &analysis,
                               std::set<RtEvent> &applied_events, bool covers);
      void update_owner(const AddressSpaceID new_logical_owner); 
      bool replicate_logical_owner_space(AddressSpaceID source, 
                             const CollectiveMapping *mapping, bool need_lock);
      void process_replication_response(AddressSpaceID owner);
      void process_replication_invalidation(std::vector<RtEvent> &applied);
    protected:
      void pack_state(Serializer &rez, const AddressSpaceID target,
            DistributedID target_did, IndexSpaceExpression *target_expr,
            IndexSpaceExpression *expr, const bool expr_covers,
            const FieldMask &mask, const bool pack_guards);
      void unpack_state_and_apply(Deserializer &derez, 
          const AddressSpaceID source, const bool forward_to_owner,
          std::vector<RtEvent> &ready_events);
      void invalidate_state(IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &mask);
      void clone_to_local(EquivalenceSet *dst, FieldMask mask,
                          IndexSpaceExpression *clone_expr,
                          std::vector<RtEvent> &applied_events,
                          const bool invalidate_overlap,
                          const bool forward_to_owner);
      void clone_to_remote(DistributedID target, AddressSpaceID target_space,
                    IndexSpaceExpression *target_expr, 
                    IndexSpaceExpression *overlap, FieldMask mask,
                    std::vector<RtEvent> &applied_events,
                    const bool invalidate_overlap, const bool forward_to_owner);
      void find_overlap_updates(IndexSpaceExpression *overlap, 
            const bool overlap_covers, const FieldMask &mask, 
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> > &valid_updates,
            FieldMaskSet<IndexSpaceExpression> &initialized_updates,
            std::map<unsigned,std::list<std::pair<InstanceView*,
                IndexSpaceExpression*> > > &reduction_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > &restricted_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > &released_updates,
            FieldMaskSet<CopyFillGuard> *read_only_guard_updates,
            FieldMaskSet<CopyFillGuard> *reduction_fill_guard_updates,
            TraceViewSet *&precondition_updates,
            TraceViewSet *&anticondition_updates,
            TraceViewSet *&postcondition_updates, 
            DistributedID target, IndexSpaceExpression *target_expr) const;
      void apply_state(LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> > &valid_updates,
            FieldMaskSet<IndexSpaceExpression> &initialized_updates,
            std::map<unsigned,std::list<std::pair<InstanceView*,
                IndexSpaceExpression*> > > &reduction_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > &restricted_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > &released_updates,
            TraceViewSet *precondition_updates,
            TraceViewSet *anticondition_updates,
            TraceViewSet *postcondition_updates,
            FieldMaskSet<CopyFillGuard> *read_only_guard_updates,
            FieldMaskSet<CopyFillGuard> *reduction_fill_guard_updates,
            std::vector<RtEvent> &applied_events,
            const bool needs_lock, const bool forward_to_owner,
            const bool unpack_references);
      static void pack_updates(Serializer &rez, const AddressSpaceID target,
            const LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> > &valid_updates,
            const FieldMaskSet<IndexSpaceExpression> &initialized_updates,
            const std::map<unsigned,std::list<std::pair<InstanceView*,
                IndexSpaceExpression*> > > &reduction_updates,
            const LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > &restricted_updates,
            const LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > &released_updates,
            const FieldMaskSet<CopyFillGuard> *read_only_updates,
            const FieldMaskSet<CopyFillGuard> *reduction_fill_updates,
            const TraceViewSet *precondition_updates,
            const TraceViewSet *anticondition_updates,
            const TraceViewSet *postcondition_updates,
            const bool pack_references);
    public:
      static void handle_make_owner(const void *args);
      static void handle_apply_state(const void *args);
    public:
      static void handle_equivalence_set_request(Deserializer &derez,
                                                 Runtime *runtime);
      static void handle_equivalence_set_response(Deserializer &derez,
                                                  Runtime *runtime);
      static void handle_migration(Deserializer &derez, 
                                   Runtime *runtime, AddressSpaceID source);
      static void handle_owner_update(Deserializer &derez, Runtime *rt);
      static void handle_make_owner(Deserializer &derez, Runtime *rt);
      static void handle_invalidate_trackers(Deserializer &derez, Runtime *rt);
      static void handle_replication_request(Deserializer &derez, Runtime *rt,
                                             AddressSpaceID source);
      static void handle_replication_response(Deserializer &derez, Runtime *rt);
      static void handle_replication_invalidation(Deserializer &derez,
                                                  Runtime *rt);
      static void handle_clone_request(Deserializer &derez, Runtime *runtime, 
                                       AddressSpaceID source);
      static void handle_clone_response(Deserializer &derez, Runtime *runtime);
      static void handle_capture_request(Deserializer &derez, Runtime *runtime,
                                         AddressSpaceID source);
      static void handle_capture_response(Deserializer &derez, Runtime *runtime,
                                          AddressSpaceID source);
    public:
      // Note this context refers to the context from which the views are
      // created in. Normally this is the same as the context in which the
      // equivalence set is made, but it can be from an ancestor task
      // higher up in the task tree in the presencce of virtual mappings.
      // It's crucial to correctness that all views stored in an equivalence
      // set come from the same context.
      InnerContext *const context;
      IndexSpaceExpression *const set_expr;
      const RegionTreeID tree_id;
    protected:
      mutable LocalLock                                 eq_lock;
      // This is the physical state of the equivalence set
      FieldMaskSet<LogicalView>                         total_valid_instances;
      typedef LegionMap<LogicalView*, FieldMaskSet<IndexSpaceExpression> > 
      ViewExprMaskSets;
      ViewExprMaskSets                                  partial_valid_instances;
      FieldMask                                         partial_valid_fields;
      // Expressions and fields that have valid data
      FieldMaskSet<IndexSpaceExpression>                initialized_data;
      // Reductions always need to be applied in order so keep them in order
      std::map<unsigned/*fidx*/,std::list<std::pair<
        InstanceView*,IndexSpaceExpression*> > >        reduction_instances;
      FieldMask                                         reduction_fields;
      // The list of expressions with the single instance for each
      // field that represents the restriction of that expression
      typedef LegionMap<IndexSpaceExpression*, FieldMaskSet<InstanceView> > 
      ExprViewMaskSets;
      ExprViewMaskSets                                  restricted_instances;
      // Summary of any field that has a restriction
      FieldMask                                         restricted_fields;
      // List of instances that were restricted, but have been acquired
      ExprViewMaskSets                                  released_instances;
      // The names of any collective views we have resident in the
      // total or partial valid instances. This allows us to do faster
      // checks for aliasing when we just have a bunch of individual views
      // and no collective views (the common case). Once you start adding
      // in collective views then some of the analyses can get more 
      // expensive but that is the trade-off with using collective views.
      FieldMaskSet<CollectiveView>                      collective_instances;
    protected:
      // Tracing state for this equivalence set
      TraceViewSet                                      *tracing_preconditions;
      TraceViewSet                                      *tracing_anticonditions;
      TraceViewSet                                      *tracing_postconditions;
    protected:
      // This tracks the most recent copy-fill aggregator for each field in 
      // read-only cases so that reads the depend on each other are ordered
      FieldMaskSet<CopyFillGuard>                       read_only_guards;
      // This tracks the most recent fill-aggregator for each field in reduction
      // cases so that reductions that depend on the same fill are ordered
      FieldMaskSet<CopyFillGuard>                       reduction_fill_guards;
      // An event to order to deferral tasks
      std::atomic<Realm::Event::id_t>                next_deferral_precondition;
    protected:
      // This node is the node which contains the valid state data
      AddressSpaceID                                    logical_owner_space;
      // To support control-replicated mapping of common regions in index space
      // launches or collective instance mappings, we need to have a way to 
      // force all analyses to see the same value for the logical owner space
      // for the equivalence set such that we can then have a single analysis
      // traverse the equivalence set without requiring communication between
      // the analyses to determine which one does the traversal. The 
      // replicated_owner_state defines a spanning tree of the existing
      // copies of the equivalence sets that all share knowledge of the
      // logical owner space.
      ReplicatedOwnerState*                             replicated_owner_state;
    protected:
      // Uses these for determining when we should do migration
      // There is an implicit assumption here that equivalence sets
      // are only used be a small number of nodes that is less than
      // the samples per migration count, if it ever exceeds this 
      // then we'll issue a warning
      static constexpr unsigned SAMPLES_PER_MIGRATION_TEST = 64;
      // How many total epochs we want to remember
      static constexpr unsigned MIGRATION_EPOCHS = 2;
      std::vector<std::pair<AddressSpaceID,unsigned> > 
        user_samples[MIGRATION_EPOCHS];
      unsigned migration_index;
      unsigned sample_count;
    };

#if 0
    /**
     * \class PendingEquivalenceSet
     * This is a helper class to store the equivalence sets for
     * pending refinements where we have computed a new refinement
     * but haven't made the equivalence set yet to represent it
     */
    class PendingEquivalenceSet : public LegionHeapify<PendingEquivalenceSet> {
    public:
      struct DeferFinalizePendingSetArgs : 
        public LgTaskArgs<DeferFinalizePendingSetArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_FINALIZE_PENDING_SET_TASK_ID;
      public:
        DeferFinalizePendingSetArgs(PendingEquivalenceSet *p)
          : LgTaskArgs<DeferFinalizePendingSetArgs>(implicit_provenance), 
            pending(p) { }
      public:
        PendingEquivalenceSet *const pending;
      };
    public:
      PendingEquivalenceSet(RegionNode *region_node, InnerContext *context);
      PendingEquivalenceSet(const PendingEquivalenceSet &rhs) = delete;
      ~PendingEquivalenceSet(void);
    public:
      PendingEquivalenceSet& operator=(const PendingEquivalenceSet&) = delete;
    public:
      void record_previous(EquivalenceSet *set, const FieldMask &mask);
      void record_all(VersionInfo &version_info); 
    public:
      EquivalenceSet* compute_refinement(AddressSpaceID suggested_owner,
                      Runtime *runtime, std::set<RtEvent> &ready_events);
      bool finalize(void);
      static void handle_defer_finalize(const void *args);
    public:
      RegionNode *const region_node;
      InnerContext *const context;
    protected:
      EquivalenceSet *new_set;
      RtEvent clone_event;
      FieldMaskSet<EquivalenceSet> previous_sets;
    };
#endif

    /**
     * \class VersionManager
     * The VersionManager class tracks the starting equivalence
     * sets for a given node in the logical region tree. Note
     * that its possible that these have since been shattered
     * and we need to traverse them, but it's a cached starting
     * point that doesn't involve tracing the entire tree.
     */
    class VersionManager : public EqSetTracker, 
                           public LegionHeapify<VersionManager> {
    public:
      static const AllocationType alloc_type = VERSION_MANAGER_ALLOC;
    public:
      struct LgFinalizeEqSetsArgs : 
        public LgTaskArgs<LgFinalizeEqSetsArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FINALIZE_EQ_SETS_TASK_ID;
      public:
        LgFinalizeEqSetsArgs(VersionManager *man, RtUserEvent c, UniqueID uid)
          : LgTaskArgs<LgFinalizeEqSetsArgs>(uid), manager(man), compute(c) { }
      public:
        VersionManager *const manager;
        const RtUserEvent compute;
      };
    public:
      struct WaitingVersionInfo {
      public:
        WaitingVersionInfo(VersionInfo *info, const FieldMask &m,
                           IndexSpaceExpression *e, bool covers)
          : version_info(info), waiting_mask(m), expr(e), expr_covers(covers) 
        { }
      public:
        VersionInfo *version_info;
        FieldMask waiting_mask;
        IndexSpaceExpression *expr;
        bool expr_covers;
      }; 
    public:
      VersionManager(RegionTreeNode *node, ContextID ctx); 
      VersionManager(const VersionManager &manager) = delete;
      virtual ~VersionManager(void);
    public:
      VersionManager& operator=(const VersionManager &rhs) = delete;
    public:
      inline bool has_versions(const FieldMask &mask) const 
        { return !(mask - equivalence_sets.get_valid_mask()); }
      inline const FieldMask& get_version_mask(void) const
        { return equivalence_sets.get_valid_mask(); }
    public:
      void perform_versioning_analysis(InnerContext *parent_ctx,
                                       VersionInfo *version_info,
                                       RegionNode *region_node,
                                       IndexSpaceExpression *expr,
                                       const bool expr_covers,
                                       const FieldMask &version_mask,
                                       UniqueID opid, unsigned parent_req_index,
                                       std::set<RtEvent> &ready);
    protected:
#if 0
      void add_node_disjoint_complete_ref(void) const;
      void remove_node_disjoint_complete_ref(void) const;
#endif
      void record_equivalence_sets(VersionInfo *version_info,
                                   const FieldMask &mask,
                                   IndexSpaceExpression *expr,
                                   const bool expr_covers,
                                   std::set<RtEvent> &ready_events) const;
    public:
      virtual void add_subscription_reference(unsigned count = 1);
      virtual bool remove_subscription_reference(unsigned count = 1);
      virtual RegionTreeID get_region_tree_id(void) const;
      virtual IndexSpaceExpression* get_tracker_expression(void) const;
      virtual size_t count_outstanding_requests(void) const;
      virtual ReferenceSource get_reference_source_kind(void) const 
        { return VERSION_MANAGER_REF; } 
    public:
      void finalize_equivalence_sets(RtUserEvent done_event);                           
      void finalize_manager(void);
    public:
#if 0
      // Call these from region nodes
      void initialize_versioning_analysis(EquivalenceSet *set,
                                          const FieldMask &mask);
      void initialize_nonexclusive_virtual_analysis(const FieldMask &mask,
                                    const FieldMaskSet<EquivalenceSet> &sets);
      void compute_equivalence_sets(IndexSpaceExpression *expr,
                                    EqSetTracker *target, 
                                    const AddressSpaceID target_space,
                                    FieldMask mask,
                                    std::set<RtEvent> &ready_events,
                                    FieldMaskSet<PartitionNode> &children,
                                    FieldMask &parent_traversal,
                                    const bool downward_only);
      void find_or_create_empty_equivalence_sets(EqSetTracker *target,
                                    const AddressSpaceID target_space,
                                    const FieldMask &mask,
                                    const AddressSpaceID source,
                                    std::set<RtEvent> &ready_events);
      static void handle_compute_equivalence_sets_response(
                  Deserializer &derez, Runtime *runtime, AddressSpaceID source);
      // Call these from region nodes
      void record_refinement(EquivalenceSet *set, const FieldMask &mask,
                             FieldMask &parent_mask);
      void record_empty_refinement(const FieldMask &mask);
      // Call these from partition nodes
      void record_refinement(ShardedColorMap *map, const FieldMask &mask,
                             FieldMask &parent_mask);
      void compute_equivalence_sets(IndexSpaceExpression *expr,
                                    const FieldMask &mask, 
                                    FieldMask &parent_traversal, 
                                    FieldMaskSet<RegionNode> &children,
                                    std::map<ShardID,
                                      LegionMap<LegionColor,
                                          FieldMask> > &shard_children,
                                    const bool downward_only,
                                    const bool expr_covers) const;
      // Call these from either type of region tree node
      void propagate_refinement(RegionTreeNode *child, 
                                const FieldMask &child_mask, 
                                FieldMask &parent_mask);
      void invalidate_refinement(InnerContext &context,
                                 const FieldMask &mask, bool invalidate_self,
                                 FieldMaskSet<RegionTreeNode> &to_traverse,
                                 LegionMap<AddressSpaceID,
                                  SubscriberInvalidations> &subscribers,
                                 std::vector<EquivalenceSet*> &to_release,
                                 bool nonexclusive_virtual_mapping_root=false);
      void merge(VersionManager &src, std::set<RegionTreeNode*> &to_traverse,
               LegionMap<AddressSpaceID,SubscriberInvalidations> &subscribers,
               const std::vector<ShardID> *shard_to_shard_mapping);
      void convert(VersionManager &src, std::set<RegionTreeNode*> &to_traverse,
               LegionMap<AddressSpaceID,SubscriberInvalidations> &subscribers,
               const std::vector<ShardID> *shard_to_shard_mapping);
      void pack_manager(Serializer &rez, const bool invalidate, 
                std::map<LegionColor,RegionTreeNode*> &to_traverse,
                LegionMap<AddressSpaceID,SubscriberInvalidations> &subscribers);
      void unpack_manager(Deserializer &derez,
                          std::map<LegionColor,RegionTreeNode*> &to_traverse);
      void filter_refinement_subscriptions(const FieldMask &mask,
               LegionMap<AddressSpaceID,SubscriberInvalidations> &subscribers);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                                TreeStateLogger *logger);
#endif
    public:
      static void handle_finalize_eq_sets(const void *args);
    public:
      const ContextID ctx;
      RegionTreeNode *const node;
      Runtime *const runtime;
    protected:
      mutable LocalLock manager_lock;
    protected: 
      LegionList<WaitingVersionInfo> waiting_infos;
      LegionMap<RtUserEvent,FieldMask> equivalence_sets_ready;
#if 0
    protected:
      // The fields for which this node has disjoint complete information
      FieldMask disjoint_complete;
      // Track which disjoint and complete children we have from this
      // node for representing the refinement tree. Note that if this
      // context is control replicated this set might not be complete
      // for partition nodes, Some sub-region nodes might only exist
      // in contexts on remote shards. In those cases we will find an
      // entry for those fields for disjoint_complete_children_shards.
      FieldMaskSet<RegionTreeNode> disjoint_complete_children;
      // This data structure tracks look-ups for children which only
      // other shards no about for control replication. This data
      // structure will only be non-empty on partition nodes
      FieldMaskSet<ShardedColorMap> disjoint_complete_children_shards;
      // Track all the equivalence set trackers that are tracking this
      // refinement so that we can invalidate them whenever this refinement
      // is invalidated. Note that we only need to record the fields that
      // each tracker is following here because there is a one-to-one mapping
      // between fields and equivalence sets in a node represeting a refinement
      LegionMap<AddressSpaceID,
                FieldMaskSet<EqSetTracker> > refinement_subscriptions;
#endif
    };

    typedef DynamicTableAllocator<VersionManager,10,8> VersionManagerAllocator; 

    /**
     * \class RegionTreePath
     * Keep track of the path and states associated with a 
     * given region requirement of an operation.
     */
    class RegionTreePath {
    public:
      RegionTreePath(void);
    public:
      void initialize(unsigned min_depth, unsigned max_depth);
      void register_child(unsigned depth, const LegionColor color);
      void clear();
    public:
#ifdef DEBUG_LEGION 
      bool has_child(unsigned depth) const;
      LegionColor get_child(unsigned depth) const;
#else
      inline bool has_child(unsigned depth) const
        { return path[depth] != INVALID_COLOR; }
      inline LegionColor get_child(unsigned depth) const
        { return path[depth]; }
#endif
      inline unsigned get_path_length(void) const
        { return ((max_depth-min_depth)+1); }
      inline unsigned get_min_depth(void) const { return min_depth; }
      inline unsigned get_max_depth(void) const { return max_depth; }
    protected:
      std::vector<LegionColor> path;
      unsigned min_depth;
      unsigned max_depth;
    };

    /**
     * \class PathTraverser
     * An abstract class which provides the needed
     * functionality for walking a path and visiting
     * all the kinds of nodes along the path.
     */
    class PathTraverser {
    public:
      PathTraverser(RegionTreePath &path);
      PathTraverser(const PathTraverser &rhs);
      virtual ~PathTraverser(void);
    public:
      PathTraverser& operator=(const PathTraverser &rhs);
    public:
      // Return true if the traversal was successful
      // or false if one of the nodes exit stopped early
      bool traverse(RegionTreeNode *start);
    public:
      virtual bool visit_region(RegionNode *node) = 0;
      virtual bool visit_partition(PartitionNode *node) = 0;
    protected:
      RegionTreePath &path;
    protected:
      // Fields are only valid during traversal
      unsigned depth;
      bool has_child;
      LegionColor next_child;
    };

    /**
     * \class NodeTraverser
     * An abstract class which provides the needed
     * functionality for visiting a node in the tree
     * and all of its sub-nodes.
     */
    class NodeTraverser {
    public:
      NodeTraverser(bool force = false)
        : force_instantiation(force) { }
    public:
      virtual bool break_early(void) const { return false; }
      virtual bool visit_only_valid(void) const = 0;
      virtual bool visit_region(RegionNode *node) = 0;
      virtual bool visit_partition(PartitionNode *node) = 0;
    public:
      const bool force_instantiation;
    };

    /**
     * \class CurrentInitializer 
     * A class for initializing current states 
     */
    class CurrentInitializer : public NodeTraverser {
    public:
      CurrentInitializer(ContextID ctx);
      CurrentInitializer(const CurrentInitializer &rhs);
      ~CurrentInitializer(void);
    public:
      CurrentInitializer& operator=(const CurrentInitializer &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
    };

    /**
     * \class CurrentInvalidator 
     * A class for invalidating current states 
     */
    class CurrentInvalidator : public NodeTraverser {
    public:
      CurrentInvalidator(ContextID ctx, bool users_only);
      CurrentInvalidator(const CurrentInvalidator &rhs);
      ~CurrentInvalidator(void);
    public:
      CurrentInvalidator& operator=(const CurrentInvalidator &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const bool users_only;
    };

    /**
     * \class DeletionInvalidator
     * A class for invalidating current states for deletions
     */
    class DeletionInvalidator : public NodeTraverser {
    public:
      DeletionInvalidator(ContextID ctx, const FieldMask &deletion_mask);
      DeletionInvalidator(const DeletionInvalidator &rhs);
      ~DeletionInvalidator(void);
    public:
      DeletionInvalidator& operator=(const DeletionInvalidator &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const FieldMask &deletion_mask;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_ANALYSIS_H__
