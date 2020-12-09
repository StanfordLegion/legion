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

#ifndef __LEGION_ANALYSIS_H__
#define __LEGION_ANALYSIS_H__

#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct GenericUser
     * A base struct for tracking the user of a logical region
     */
    struct GenericUser {
    public:
      GenericUser(void) { }
      GenericUser(const RegionUsage &u, const FieldMask &m)
        : usage(u), field_mask(m) { }
    public:
      RegionUsage usage;
      FieldMask field_mask;
    };

    /**
     * \struct LogicalUser
     * A class for representing logical users of a logical 
     * region including the necessary information to
     * register mapping dependences on the user.
     */
    struct LogicalUser : public GenericUser {
    public:
      LogicalUser(void);
      LogicalUser(Operation *o, unsigned id, 
                  const RegionUsage &u, const FieldMask &m);
      LogicalUser(Operation *o, GenerationID gen, unsigned id,
                  const RegionUsage &u, const FieldMask &m);
    public:
      Operation *op;
      unsigned idx;
      GenerationID gen;
      // This field addresses a problem regarding when
      // to prune tasks out of logical region tree data
      // structures.  If no later task ever performs a
      // dependence test against this user, we might
      // never prune it from the list.  This timeout
      // prevents that from happening by forcing a
      // test to be performed whenever the timeout
      // reaches zero.
      int timeout;
#ifdef LEGION_SPY
      UniqueID uid;
#endif
    public:
      static const int TIMEOUT = LEGION_DEFAULT_LOGICAL_USER_TIMEOUT;
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
      LegionTrace *const trace;
      const unsigned req_idx;
      const RegionRequirement &req;
      const bool already_traced;
      const bool recording_trace;
      const bool replaying_trace;
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
          std::set<RtEvent> &applied, const AddressSpaceID target) = 0; 
      virtual RtEvent get_collect_event(void) const = 0;
      virtual PhysicalTraceRecorder* clone(Memoizable *memo) { return this; }
    public:
      virtual void record_get_term_event(Memoizable *memo) = 0;
      virtual void request_term_event(ApUserEvent &term_event) = 0;
      virtual void record_create_ap_user_event(ApUserEvent lhs, 
                                               Memoizable *memo) = 0;
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        Memoizable *memo) = 0;
    public:
      virtual void record_merge_events(ApEvent &lhs, 
                                       ApEvent rhs, Memoizable *memo) = 0;
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, 
                                       ApEvent e2, Memoizable *memo) = 0;
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       ApEvent e3, Memoizable *memo) = 0;
      virtual void record_merge_events(ApEvent &lhs, 
                            const std::set<ApEvent>& rhs, Memoizable *memo) = 0;
    public:
      virtual void record_issue_copy(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           ReductionOpID redop, bool reduction_fold) = 0;
      virtual void record_issue_indirect(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
                           const std::vector<void*> &indirections,
                           ApEvent precondition, PredEvent pred_guard) = 0;
      virtual void record_copy_views(ApEvent lhs, IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied) = 0;
      virtual void record_issue_fill(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard) = 0;
#ifdef LEGION_GPU_REDUCTIONS
      virtual void record_gpu_reduction(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<CopySrcDstField> &dst_fields,
                           Processor gpu, TaskID gpu_task_id,
                           PhysicalManager *src, PhysicalManager *dst,
                           ApEvent precondition, PredEvent pred_guard,
                           ReductionOpID redop, bool reduction_fold) = 0;
#endif
      virtual void record_fill_views(ApEvent lhs, IndexSpaceExpression *expr,
                           const FieldMaskSet<FillView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied_events,
                           const bool reduction_initialization) = 0;
    public:
      virtual void record_op_view(Memoizable *memo,
                          unsigned idx,
                          InstanceView *view,
                          RegionNode *node,
                          const RegionUsage &usage,
                          const FieldMask &user_mask,
                          bool update_validity,
                          std::set<RtEvent> &applied) = 0;
      virtual void record_set_op_sync_event(ApEvent &lhs, Memoizable *memo) = 0;
      virtual void record_mapper_output(Memoizable *memo,
                         const Mapper::MapTaskOutput &output,
                         const std::deque<InstanceSet> &physical_instances,
                         std::set<RtEvent> &applied_events) = 0;
      virtual void record_set_effects(Memoizable *memo, ApEvent &rhs) = 0;
      virtual void record_complete_replay(Memoizable *memo, ApEvent rhs) = 0;
      virtual void record_reservations(Memoizable *memo, ApEvent &lhs,
                              const std::map<Reservation,bool> &locks, 
                              ApEvent precondition, ApEvent postcondition) = 0;
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
        REMOTE_TRACE_RECORD_GET_TERM,
        REMOTE_TRACE_REQUEST_TERM_EVENT,
        REMOTE_TRACE_CREATE_USER_EVENT,
        REMOTE_TRACE_TRIGGER_EVENT,
        REMOTE_TRACE_MERGE_EVENTS,
        REMOTE_TRACE_ISSUE_COPY,
        REMOTE_TRACE_COPY_VIEWS,
        REMOTE_TRACE_ISSUE_FILL,
        REMOTE_TRACE_FILL_VIEWS,
        REMOTE_TRACE_RECORD_OP_VIEW,
        REMOTE_TRACE_SET_OP_SYNC,
        REMOTE_TRACE_SET_EFFECTS,
        REMOTE_TRACE_RECORD_MAPPER_OUTPUT,
        REMOTE_TRACE_COMPLETE_REPLAY,
        REMOTE_TRACE_ACQUIRE_RELEASE,
#ifdef LEGION_GPU_REDUCTIONS
        REMOTE_TRACE_GPU_REDUCTION,
#endif
      };
    public:
      RemoteTraceRecorder(Runtime *rt, AddressSpaceID origin,AddressSpace local,
                          Memoizable *memo, PhysicalTemplate *tpl, 
                          RtUserEvent applied_event, RtEvent collect_event);
      RemoteTraceRecorder(const RemoteTraceRecorder &rhs);
      virtual ~RemoteTraceRecorder(void);
    public:
      RemoteTraceRecorder& operator=(const RemoteTraceRecorder &rhs);
    public:
      virtual bool is_recording(void) const { return true; }
      virtual void add_recorder_reference(void);
      virtual bool remove_recorder_reference(void);
      virtual void pack_recorder(Serializer &rez, 
          std::set<RtEvent> &applied, const AddressSpaceID target);
      virtual RtEvent get_collect_event(void) const { return collect_event; }
      virtual PhysicalTraceRecorder* clone(Memoizable *memo);
    public:
      virtual void record_get_term_event(Memoizable *memo);
      virtual void request_term_event(ApUserEvent &term_event);
      virtual void record_create_ap_user_event(ApUserEvent lhs, 
                                               Memoizable *memo);
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        Memoizable *memo);
    public:
      virtual void record_merge_events(ApEvent &lhs, 
                                       ApEvent rhs, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, 
                                       ApEvent e2, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       ApEvent e3, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, 
                            const std::set<ApEvent>& rhs, Memoizable *memo);
    public:
      virtual void record_issue_copy(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           ReductionOpID redop, bool reduction_fold);
      virtual void record_issue_indirect(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
                           const std::vector<void*> &indirections,
                           ApEvent precondition, PredEvent pred_guard);
      virtual void record_copy_views(ApEvent lhs, IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied);
      virtual void record_issue_fill(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
#ifdef LEGION_GPU_REDUCTIONS
      virtual void record_gpu_reduction(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<CopySrcDstField> &dst_fields,
                           Processor gpu, TaskID gpu_task_id,
                           PhysicalManager *src, PhysicalManager *dst,
                           ApEvent precondition, PredEvent pred_guard,
                           ReductionOpID redop, bool reduction_fold);
#endif
      virtual void record_fill_views(ApEvent lhs, IndexSpaceExpression *expr,
                           const FieldMaskSet<FillView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied_events,
                           const bool reduction_initialization);
    public:
      virtual void record_op_view(Memoizable *memo,
                          unsigned idx,
                          InstanceView *view,
                          RegionNode *node,
                          const RegionUsage &usage,
                          const FieldMask &user_mask,
                          bool update_validity,
                          std::set<RtEvent> &applied);
      virtual void record_set_op_sync_event(ApEvent &lhs, Memoizable *memo);
      virtual void record_mapper_output(Memoizable *memo,
                          const Mapper::MapTaskOutput &output,
                          const std::deque<InstanceSet> &physical_instances,
                          std::set<RtEvent> &applied_events);
      virtual void record_set_effects(Memoizable *memo, ApEvent &rhs);
      virtual void record_complete_replay(Memoizable *memo, ApEvent rhs);
      virtual void record_reservations(Memoizable *memo, ApEvent &lhs,
                              const std::map<Reservation,bool> &locks,
                              ApEvent precondition, ApEvent postcondition);
    public:
      static RemoteTraceRecorder* unpack_remote_recorder(Deserializer &derez,
                                          Runtime *runtime, Memoizable *memo);
      static void handle_remote_update(Deserializer &derez, 
                  Runtime *runtime, AddressSpaceID source);
      static void handle_remote_response(Deserializer &derez);
    protected:
      static void pack_src_dst_field(Serializer &rez, const CopySrcDstField &f);
      static void unpack_src_dst_field(Deserializer &derez, CopySrcDstField &f);
    protected:
      Runtime *const runtime;
      const AddressSpaceID origin_space;
      const AddressSpaceID local_space;
      Memoizable *const memoizable;
      PhysicalTemplate *const remote_tpl;
      const RtUserEvent applied_event;
      mutable LocalLock applied_lock;
      std::set<RtEvent> applied_events;
      const RtEvent collect_event;
    };

    /**
     * \struct TraceInfo
     */
    struct TraceInfo {
    public:
      explicit TraceInfo(Operation *op, bool initialize = false);
      TraceInfo(const TraceInfo &info);
      TraceInfo(const TraceInfo &info, Operation *op);
      ~TraceInfo(void);
    protected:
      TraceInfo(Operation *op, Memoizable *memo, 
                PhysicalTraceRecorder *rec, bool recording);
    public:
      void pack_remote_trace_info(Serializer &rez, AddressSpaceID target,
                                  std::set<RtEvent> &applied) const;
      static TraceInfo* unpack_remote_trace_info(Deserializer &derez,
                                    Operation *op, Runtime *runtime);
      TraceInfo* clone(Operation *op);
    public:
      inline void record_get_term_event(void) const
        {
          base_sanity_check();
          rec->record_get_term_event(memo);
        }
      inline void request_term_event(ApUserEvent &term_event)
        {
          base_sanity_check();
          rec->request_term_event(term_event);
        }
      inline void record_create_ap_user_event(ApUserEvent result) const
        {
          base_sanity_check();
          rec->record_create_ap_user_event(result, memo);
        }
      inline void record_trigger_event(ApUserEvent result, ApEvent rhs) const
        {
          base_sanity_check();
          rec->record_trigger_event(result, rhs, memo);
        }
      inline void record_merge_events(ApEvent &result, 
                                      ApEvent e1, ApEvent e2) const
        {
          base_sanity_check();
          rec->record_merge_events(result, e1, e2, memo);
        }
      inline void record_merge_events(ApEvent &result, ApEvent e1, 
                                      ApEvent e2, ApEvent e3) const
        {
          base_sanity_check();
          rec->record_merge_events(result, e1, e2, e3, memo);
        }
      inline void record_merge_events(ApEvent &result, 
                                      const std::set<ApEvent> &events) const
        {
          base_sanity_check();
          rec->record_merge_events(result, events, memo);
        }
      inline void record_op_sync_event(ApEvent &result) const
        {
          base_sanity_check();
          rec->record_set_op_sync_event(result, memo);
        }
      inline void record_mapper_output(Memoizable *local, 
                          const Mapper::MapTaskOutput &output,
                          const std::deque<InstanceSet> &physical_instances,
                          std::set<RtEvent> &applied)
        {
          base_sanity_check();
          rec->record_mapper_output(local, output, physical_instances, applied);
        }
      inline void record_set_effects(Memoizable *memo, ApEvent &rhs) const
        {
          base_sanity_check();
          rec->record_set_effects(memo, rhs);
        }
      inline void record_complete_replay(Memoizable *local, 
                                         ApEvent ready_event) const
        {
          base_sanity_check();
          rec->record_complete_replay(local, ready_event);
        }
      inline void record_reservations(Memoizable *memo, ApEvent &lhs,
                      const std::map<Reservation,bool> &reservations,
                      ApEvent precondition, ApEvent postcondition) const
        {
          base_sanity_check();
          rec->record_reservations(memo, lhs, reservations, 
                                   precondition, postcondition);
        }
    public:
      inline RtEvent get_collect_event(void) const 
        {
          if ((memo == NULL) || !recording)
            return RtEvent::NO_RT_EVENT;
          else
            return rec->get_collect_event();
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
    public:
      Operation *const op;
      Memoizable *const memo;
    protected:
      PhysicalTraceRecorder *const rec;
    public:
      const bool recording;
    };

    /**
     * \struct PhysicalTraceInfo
     */
    struct PhysicalTraceInfo : public TraceInfo {
    public:
      PhysicalTraceInfo(Operation *op, unsigned index, bool init);
      PhysicalTraceInfo(const TraceInfo &info, unsigned index, 
                        bool update_validity = true);
      // Weird argument order to help the compiler avoid ambiguity
      PhysicalTraceInfo(unsigned src_idx, const TraceInfo &info, 
                        unsigned dst_idx);
      PhysicalTraceInfo(const PhysicalTraceInfo &rhs);
    protected:
      PhysicalTraceInfo(Operation *op, Memoizable *memo, unsigned src_idx, 
          unsigned dst_idx, bool update_validity, PhysicalTraceRecorder *rec);
    public:
      inline void record_issue_copy(ApEvent &result,
                          IndexSpaceExpression *expr,
                          const std::vector<CopySrcDstField>& src_fields,
                          const std::vector<CopySrcDstField>& dst_fields,
#ifdef LEGION_SPY
                          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                          ApEvent precondition, PredEvent pred_guard,
                          ReductionOpID redop, bool reduction_fold) const
        {
          sanity_check();
          rec->record_issue_copy(memo, result, expr, src_fields, dst_fields,
#ifdef LEGION_SPY
                                 src_tree_id, dst_tree_id,
#endif
                                 precondition, pred_guard,redop,reduction_fold);
        }
      inline void record_issue_fill(ApEvent &result,
                          IndexSpaceExpression *expr,
                          const std::vector<CopySrcDstField> &fields,
                          const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                          FieldSpace handle,
                          RegionTreeID tree_id,
#endif
                          ApEvent precondition, PredEvent pred_guard) const
        {
          sanity_check();
          rec->record_issue_fill(memo, result, expr, fields, 
                                 fill_value, fill_size,
#ifdef LEGION_SPY
                                 handle, tree_id,
#endif
                                 precondition, pred_guard);
        }
#ifdef LEGION_GPU_REDUCTIONS
      inline void record_gpu_reduction(ApEvent &result,
                                IndexSpaceExpression *expr,
                                const std::vector<CopySrcDstField> &src_fields,
                                const std::vector<CopySrcDstField> &dst_fields,
                                Processor gpu, TaskID gpu_task_id,
                                PhysicalManager *src, PhysicalManager *dst,
                                ApEvent precondition, PredEvent pred_guard,
                                ReductionOpID redop, bool reduction_fold) const
        {
          sanity_check();
          rec->record_gpu_reduction(memo, result, expr, src_fields, dst_fields,
                                    gpu, gpu_task_id, src, dst, precondition, 
                                    pred_guard, redop, reduction_fold);
        }
#endif
      inline void record_fill_views(ApEvent lhs,
                                    IndexSpaceExpression *expr,
                                    const FieldMaskSet<FillView> &srcs,
                                    const FieldMaskSet<InstanceView> &dsts,
                                    std::set<RtEvent> &applied,
                                    const bool reduction_initialization) const
        {
          sanity_check();
          rec->record_fill_views(lhs, expr, srcs, dsts, applied,
                                 reduction_initialization);
        }
      inline void record_issue_indirect(ApEvent &result,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
                             const std::vector<void*> &indirections,
                             ApEvent precondition, PredEvent pred_guard) const
        {
          sanity_check();
          rec->record_issue_indirect(memo, result, expr, src_fields, dst_fields,
                                     indirections, precondition, pred_guard);
        }
      inline void record_copy_views(ApEvent lhs, IndexSpaceExpression *expr,
                                 const FieldMaskSet<InstanceView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                    std::set<RtEvent> &applied) const
        {
          sanity_check();
          rec->record_copy_views(lhs, expr, tracing_srcs, tracing_dsts,applied);
        }
      inline void record_op_view(const RegionUsage &usage,
                                 const FieldMask &user_mask,
                                 InstanceView *view, RegionNode *node,
                                 std::set<RtEvent> &applied) const
        {
          sanity_check();
          rec->record_op_view(memo, index, view, node, usage, user_mask,
                              update_validity, applied);
        }
    public:
      template<bool PACK_OPERATION>
      void pack_trace_info(Serializer &rez, std::set<RtEvent> &applied,
                           const AddressSpaceID target) const;
      static PhysicalTraceInfo unpack_trace_info(Deserializer &derez,
              Runtime *runtime, std::set<RtEvent> &ready_events);
      static PhysicalTraceInfo unpack_trace_info(Deserializer &derez,
                                     Runtime *runtime, Operation *op);
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
#ifdef ENABLE_VIEW_REPLICATION
      PhysicalUser(const RegionUsage &u, IndexSpaceExpression *expr,
                   UniqueID op_id, unsigned index, RtEvent collect_event,
                   bool copy, bool covers);
#else
      PhysicalUser(const RegionUsage &u, IndexSpaceExpression *expr,
                   UniqueID op_id, unsigned index, bool copy, bool covers);
#endif
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
#ifdef ENABLE_VIEW_REPLICATION
      const RtEvent collect_event;
#endif
      const bool copy_user; // is this from a copy or an operation
      const bool covers; // whether the expr covers the ExprView its in
    };  

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
                        IndexSpaceNode *sd,
                        std::set<RtEvent> &applied);
      ProjectionSummary(const ProjectionInfo &info, std::set<RtEvent> &applied);
      ProjectionSummary(const ProjectionSummary &rhs);
      ~ProjectionSummary(void);
    public:
      ProjectionSummary& operator=(const ProjectionSummary &rhs);
    public:
      bool operator<(const ProjectionSummary &rhs) const;
      bool operator==(const ProjectionSummary &rhs) const;
      bool operator!=(const ProjectionSummary &rhs) const;
    public:
      void pack_summary(Serializer &rez,
                        std::vector<DistributedCollectable*> &to_remove) const;
      static ProjectionSummary unpack_summary(Deserializer &derez,
                        RegionTreeForest *context, std::set<RtEvent> &applied);
    public:
      IndexSpaceNode *domain;
      ProjectionFunction *projection;
      ShardingFunction *sharding;
      IndexSpaceNode *sharding_domain;
    };

    /**
     * \struct FieldState
     * Track the field state more accurately
     * for logical traversals to figure out 
     * which tasks can run in parallel.
     */
    struct FieldState {
    public:
      FieldState(void);
      FieldState(const GenericUser &u, const FieldMask &m, 
                 RegionTreeNode *child, std::set<RtEvent> &applied);
      FieldState(const RegionUsage &u, const FieldMask &m,
                 ProjectionFunction *proj, IndexSpaceNode *proj_space, 
                 ShardingFunction *sharding_function, 
                 IndexSpaceNode *sharding_space,
                 std::set<RtEvent> &applied,
                 RegionTreeNode *node, bool dirty_reduction = false);
      FieldState(const FieldState &rhs);
      FieldState(FieldState &&rhs) noexcept;
      FieldState& operator=(const FieldState &rhs);
      FieldState& operator=(FieldState &&rhs) noexcept;
      ~FieldState(void);
    public:
      inline bool is_projection_state(void) const 
        { return (open_state >= OPEN_READ_ONLY_PROJ); } 
      inline const FieldMask& valid_fields(void) const 
        { return open_children.get_valid_mask(); }
    public:
      bool overlaps(const FieldState &rhs) const;
      bool projections_match(const FieldState &rhs) const;
      void merge(FieldState &rhs, RegionTreeNode *node);
      bool filter(const FieldMask &mask);
      void add_child(RegionTreeNode *child,
          const FieldMask &mask, std::set<RtEvent> &applied);
      void remove_child(RegionTreeNode *child);
    public:
      bool can_elide_close_operation(Operation *op, unsigned index,
                                     const ProjectionInfo &info,
                                     RegionTreeNode *node, bool reduction,
                                     std::set<RtEvent> &applied_events) const;
      void record_projection_summary(const ProjectionInfo &info,
                                     RegionTreeNode *node,
                                     std::set<RtEvent> &applied_events);
    protected:
      bool expensive_elide_test(Operation *op, unsigned index,
                                const ProjectionInfo &info,
                                RegionTreeNode *node, bool reduction) const;
    public:
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask,
                       RegionNode *node) const;
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask,
                       PartitionNode *node) const;
    public:
      FieldMaskSet<RegionTreeNode> open_children;
      OpenState open_state;
      ReductionOpID redop;
      std::set<ProjectionSummary> projections;
      bool disjoint_shallow;
    };

    /**
     * \class ProjectionTree
     * This is a tree that stores the summary of a region
     * tree that is accessed by an index launch and which
     * node owns the leaves for in the case of control replication
     */
    class ProjectionTree {
    public:
      ProjectionTree(IndexTreeNode *source,
                     ShardID owner_shard = 0);
      ProjectionTree(const ProjectionTree &rhs);
      ~ProjectionTree(void);
    public:
      ProjectionTree& operator=(const ProjectionTree &rhs);
    public:
      void add_child(ProjectionTree *child);
      bool dominates(const ProjectionTree *other) const;
      bool disjoint(const ProjectionTree *other) const;
      bool all_same_shard(ShardID other_shard) const;
    public:
      IndexTreeNode *const node;
      const ShardID owner_shard;
      std::map<IndexTreeNode*,ProjectionTree*> children;
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
      LogicalState(const LogicalState &state);
      ~LogicalState(void);
    public:
      LogicalState& operator=(const LogicalState &rhs);
    public:
      void check_init(void);
      void clear_logical_users(void);
      void reset(void);
      void clear_deleted_state(const FieldMask &deleted_mask);
      void merge(LogicalState &src, std::set<RegionTreeNode*> &to_traverse);
      void swap(LogicalState &src, std::set<RegionTreeNode*> &to_traverse);
    public:
      RegionTreeNode *const owner;
    public:
      LegionList<FieldState,
                 LOGICAL_FIELD_STATE_ALLOC>::track_aligned field_states;
      LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned 
                                                            curr_epoch_users;
      LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned 
                                                            prev_epoch_users;
    public:
      // Keep track of which fields we've done a reduction to here
      FieldMask reduction_fields;
      LegionMap<ReductionOpID,FieldMask>::aligned outstanding_reductions;
    public:
      // Track whether this node is part of the disjoint-complete tree
      FieldMask disjoint_complete_tree;
      // Use this data structure for tracking where the disjoint-complete
      // tree is for this region tree. On region nodes there should be at
      // most one child in this data structure. On partition nodes there
      // can be any number of children with different field masks.
      // Note that this might also be empty for partition nodes where
      // we have issued projections
      FieldMaskSet<RegionTreeNode> disjoint_complete_children;
      // Track the names of children which have "complete writes" open
      // below. If we've written all our open children below then we
      // ourselves are written and we should propagate this information
      // up the tree to see if we want to change refinements
      FieldMaskSet<RegionTreeNode> written_disjoint_complete_children;
    };

    typedef DynamicTableAllocator<LogicalState,10,8> LogicalStateAllocator;

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
      LogicalCloser(const LogicalCloser &rhs);
      ~LogicalCloser(void);
    public:
      LogicalCloser& operator=(const LogicalCloser &rhs);
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
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
      void update_state(LogicalState &state);
      void register_close_operations(
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users);
    protected:
      void register_dependences(CloseOp *close_op, 
                                const LogicalUser &close_user,
                                const LogicalUser &current, 
                                const FieldMask &open_below,
             LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned &husers,
             LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &ausers,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
    public:
      const ContextID ctx;
      const LogicalUser &user;
      RegionTreeNode *const root_node;
      const bool validates;
      LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned closed_users;
    protected:
      FieldMask close_mask;
    protected:
      // At most we will ever generate one close operation at a node
      MergeCloseOp *close_op;
    protected:
      // Cache the generation IDs so we can kick off ops before adding users
      GenerationID merge_close_gen;
    }; 

#if NEWEQ 
    class KDTree {
    public:
      virtual ~KDTree(void) { }
      virtual bool refine(std::vector<EquivalenceSet*> &subsets,
          const FieldMask &refinement_mask, unsigned max_depth) = 0;
    };

    /**
     * \class KDNode
     * A KDNode is used for constructing a KD tree in order to divide up the 
     * sub equivalence sets when there are too many to fan out from just 
     * one parent equivalence set.
     */
    template<int DIM>
    class KDNode : public KDTree {
    public:
      KDNode(IndexSpaceExpression *expr, Runtime *runtime, 
             int refinement_dim, int last_changed_dim = -1); 
      KDNode(const Rect<DIM> &rect, Runtime *runtime,
             int refinement_dim, int last_changed_dim = -1);
      KDNode(const KDNode<DIM> &rhs);
      virtual ~KDNode(void);
    public:
      KDNode<DIM>& operator=(const KDNode<DIM> &rhs);
    public:
      virtual bool refine(std::vector<EquivalenceSet*> &subsets,
                          const FieldMask &refinement_mask, unsigned max_depth);
    public:
      static Rect<DIM> get_bounds(IndexSpaceExpression *expr);
    public:
      Runtime *const runtime;
      const Rect<DIM> bounds;
      const int refinement_dim;
      // For detecting non convext cases where we kind find a 
      // splitting plane in any dimension
      const int last_changed_dim;
    };
#endif

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef : public LegionHeapify<InstanceRef> {
    public:
      InstanceRef(bool composite = false);
      InstanceRef(const InstanceRef &rhs);
      InstanceRef(InstanceManager *manager, const FieldMask &valid_fields,
                  ApEvent ready_event = ApEvent::NO_AP_EVENT);
      ~InstanceRef(void);
    public:
      InstanceRef& operator=(const InstanceRef &rhs);
    public:
      bool operator==(const InstanceRef &rhs) const;
      bool operator!=(const InstanceRef &rhs) const;
    public:
      inline bool has_ref(void) const { return (manager != NULL); }
      inline ApEvent get_ready_event(void) const { return ready_event; }
      inline void set_ready_event(ApEvent ready) { ready_event = ready; }
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
      void add_valid_reference(ReferenceSource source, 
                               ReferenceMutator *mutator) const;
      void remove_valid_reference(ReferenceSource source,
                                  ReferenceMutator *mutator) const;
    public:
      Memory get_memory(void) const;
      PhysicalManager* get_instance_manager(void) const;
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
      ApEvent ready_event;
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
        LegionVector<InstanceRef>::aligned vector; 
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
      void add_valid_references(ReferenceSource source,
                                ReferenceMutator *mutator) const;
      void remove_valid_references(ReferenceSource source,
                                   ReferenceMutator *mutator) const;
    public:
      void update_wait_on_events(std::set<ApEvent> &wait_on_events) const;
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
    class CopyFillAggregator : public WrapperReferenceMutator, 
                               public CopyFillGuard,
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
                      ApEvent p, const bool src, const bool dst, UniqueID uid,
                      unsigned ps, const bool need_pass_pre)
          : LgTaskArgs<CopyFillAggregation>(uid), PhysicalTraceInfo(i),
            aggregator(a), pre(p), pass(ps), has_src(src), 
            has_dst(dst), need_pass_preconditions(need_pass_pre) 
          // This is kind of scary, Realm is about to make a copy of this
          // without our knowledge, but we need to preserve the correctness
          // of reference counting on PhysicalTraceRecorders, so just add
          // an extra reference here that we will remove when we're handled.
          { if (rec != NULL) rec->add_recorder_reference(); }
      public:
        inline void remove_recorder_reference(void) const
          { if ((rec != NULL) && rec->remove_recorder_reference()) delete rec; }
      public:
        CopyFillAggregator *const aggregator;
        const ApEvent pre;
        const unsigned pass;
        const bool has_src;
        const bool has_dst;
        const bool need_pass_preconditions;
      }; 
    public:
      typedef LegionMap<InstanceView*,
               FieldMaskSet<IndexSpaceExpression> >::aligned InstanceFieldExprs;
      typedef LegionMap<ApEvent,FieldMask>::aligned EventFieldMap;
      class CopyUpdate;
      class FillUpdate;
      class Update {
      public:
        Update(IndexSpaceExpression *exp, const FieldMask &mask,
               CopyAcrossHelper *helper)
          : expr(exp), src_mask(mask), across_helper(helper) { }
        virtual ~Update(void) { }
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const = 0;
        virtual void compute_source_preconditions(
                       RegionTreeForest *forest, const FieldMask &src_mask,
                       const std::map<InstanceView*,EventFieldMap> &src_pre,
                       std::set<ApEvent> &preconditions) const = 0;
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
        CopyUpdate(InstanceView *src, const FieldMask &mask,
                   IndexSpaceExpression *expr,
                   ReductionOpID red = 0,
                   CopyAcrossHelper *helper = NULL)
          : Update(expr, mask, helper), source(src), redop(red) { }
        virtual ~CopyUpdate(void) { }
      private:
        CopyUpdate(const CopyUpdate &rhs)
          : Update(rhs.expr, rhs.src_mask, rhs.across_helper), 
            source(rhs.source), redop(rhs.redop) { assert(false); }
        CopyUpdate& operator=(const CopyUpdate &rhs)
          { assert(false); return *this; }
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const;
        virtual void compute_source_preconditions(
                       RegionTreeForest *forest, const FieldMask &src_mask,
                       const std::map<InstanceView*,EventFieldMap> &src_pre,
                       std::set<ApEvent> &preconditions) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        InstanceView *const source;
        const ReductionOpID redop;
      };
      class FillUpdate : public Update, public LegionHeapify<FillUpdate> {
      public:
        FillUpdate(FillView *src, const FieldMask &mask,
                   IndexSpaceExpression *expr,
                   CopyAcrossHelper *helper = NULL)
          : Update(expr, mask, helper), source(src) { }
        virtual ~FillUpdate(void) { }
      private:
        FillUpdate(const FillUpdate &rhs)
          : Update(rhs.expr, rhs.src_mask, rhs.across_helper),
            source(rhs.source) { assert(false); }
        FillUpdate& operator=(const FillUpdate &rhs)
          { assert(false); return *this; }
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const;
        virtual void compute_source_preconditions(
                       RegionTreeForest *forest, const FieldMask &src_mask,
                       const std::map<InstanceView*,EventFieldMap> &src_pre,
                       std::set<ApEvent> &preconditions) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        FillView *const source;
      };
      typedef LegionMap<ApEvent,
               FieldMaskSet<Update> >::aligned EventFieldUpdates;
    public:
      CopyFillAggregator(RegionTreeForest *forest, Operation *op, unsigned idx,
                         RtEvent guard_event, bool track_events,
                         PredEvent pred_guard = PredEvent::NO_PRED_EVENT);
      CopyFillAggregator(RegionTreeForest *forest, Operation *op, 
                         unsigned src_idx, unsigned dst_idx,
                         RtEvent guard_event, bool track_events,
                         PredEvent pred_guard = PredEvent::NO_PRED_EVENT);
      CopyFillAggregator(const CopyFillAggregator &rhs);
      virtual ~CopyFillAggregator(void);
    public:
      CopyFillAggregator& operator=(const CopyFillAggregator &rhs);
    public:
      void record_update(InstanceView *dst_view,
                          LogicalView *src_view,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          EquivalenceSet *tracing_eq,
                          std::set<RtEvent> &applied,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      void record_updates(InstanceView *dst_view, 
                          const FieldMaskSet<LogicalView> &src_views,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          EquivalenceSet *tracing_eq,
                          std::set<RtEvent> &applied,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      void record_partial_updates(InstanceView *dst_view,
                          const LegionMap<LogicalView*,
                      FieldMaskSet<IndexSpaceExpression> >::aligned &src_views,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          EquivalenceSet *tracing_eq,
                          std::set<RtEvent> &applied,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      // Neither fills nor reductions should have a redop across as they
      // should have been applied an instance directly for across copies
      void record_fill(InstanceView *dst_view,
                       FillView *src_view,
                       const FieldMask &fill_mask,
                       IndexSpaceExpression *expr,
                       EquivalenceSet *tracing_eq,
                       std::set<RtEvent> &applied,
                       CopyAcrossHelper *across_helper = NULL);
      void record_reductions(InstanceView *dst_view,
                             const std::list<std::pair<ReductionView*,
                                    IndexSpaceExpression*> > &src_views,
                             const unsigned src_fidx,
                             const unsigned dst_fidx,
                             EquivalenceSet *tracing_eq,
                             std::set<RtEvent> &applied,
                             CopyAcrossHelper *across_helper = NULL);
      // Record preconditions coming back from analysis on views
      void record_preconditions(InstanceView *view, bool reading,
                                EventFieldMap &preconditions);
      void record_precondition(InstanceView *view, bool reading,
                               ApEvent event, const FieldMask &mask);
      void issue_updates(const PhysicalTraceInfo &trace_info, 
                         ApEvent precondition,
                         // Next two flags are used for across-copies
                         // to indicate when we already know preconditions
                         const bool has_src_preconditions = false,
                         const bool has_dst_preconditions = false,
                         const bool need_deferral = false, unsigned pass = 0, 
                         bool need_pass_preconditions = true);
      ApEvent summarize(const PhysicalTraceInfo &trace_info) const;
    protected:
      void record_view(LogicalView *new_view);
      void resize_reductions(size_t new_size);
      void update_tracing_valid_views(EquivalenceSet *tracing_eq,
            LogicalView *src, LogicalView *dst, const FieldMask &mask,
            IndexSpaceExpression *expr, ReductionOpID redop,
            std::set<RtEvent> &applied_events) const;
      RtEvent perform_updates(const LegionMap<InstanceView*,
                            FieldMaskSet<Update> >::aligned &updates,
                           const PhysicalTraceInfo &trace_info,
                           const ApEvent all_precondition, int redop_index,
                           const bool has_src_preconditions,
                           const bool has_dst_preconditions,
                           const bool needs_preconditions);
      void find_reduction_preconditions(InstanceView *dst_view, 
                           const PhysicalTraceInfo &trace_info,
                           IndexSpaceExpression *copy_expr,
                           const FieldMask &copy_mask, 
                           UniqueID op_id, unsigned redop_index, 
                           std::set<RtEvent> &preconditions_ready);
      void issue_fills(InstanceView *target,
                       const std::vector<FillUpdate*> &fills,
                       ApEvent precondition, const FieldMask &fill_mask,
                       const PhysicalTraceInfo &trace_info,
                       const bool has_dst_preconditions);
      void issue_copies(InstanceView *target, 
                        const std::map<InstanceView*,
                                       std::vector<CopyUpdate*> > &copies,
                        ApEvent precondition, const FieldMask &copy_mask,
                        const PhysicalTraceInfo &trace_info,
                        const bool has_dst_preconditions);
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
      Operation *const op;
      const unsigned src_index;
      const unsigned dst_index;
      const RtEvent guard_precondition;
      const PredEvent predicate_guard;
      const bool track_events;
    protected:
      FieldMask update_fields;
      LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned sources; 
      std::vector</*vector over reduction epochs*/
        LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned> reductions;
      // Figure out the reduction operator is for each epoch of a
      // given destination instance and field
      std::map<std::pair<InstanceView*,unsigned/*dst fidx*/>,
               std::vector<ReductionOpID> > reduction_epochs;
      std::set<LogicalView*> all_views; // used for reference counting
    protected:
      mutable LocalLock pre_lock; 
      std::map<InstanceView*,LegionMap<ApEvent,FieldMask>::aligned> 
        dst_pre, src_pre;
    protected:
      // Runtime mapping effects that we create
      std::set<RtEvent> effects; 
      // Events for the completion of our copies if we are supposed
      // to be tracking them
      std::set<ApEvent> events;
    protected:
      struct SourceQuery {
      public:
        SourceQuery(void) { }
        SourceQuery(const std::set<InstanceView*> srcs,
                    const FieldMask src_mask,
                    InstanceView *res)
          : sources(srcs), query_mask(src_mask), result(res) { }
      public:
        std::set<InstanceView*> sources;
        FieldMask query_mask;
        InstanceView *result;
      };
      // Cached calls to the mapper for selecting sources
      std::map<InstanceView*,LegionVector<SourceQuery>::aligned> mapper_queries;
    protected:
      // Help for tracing 
      FieldMaskSet<FillView> *tracing_src_fills;
      FieldMaskSet<InstanceView> *tracing_srcs;
      FieldMaskSet<InstanceView> *tracing_dsts;
    };

    /**
     * \class PhysicalAnalysis
     * This is the virtual base class for handling all traversals over 
     * equivalence set trees to perform physical analyses
     */
    class PhysicalAnalysis : public Collectable, public LocalLock {
    public:
      struct DeferPerformTraversalArgs :
        public LgTaskArgs<DeferPerformTraversalArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_TRAVERSAL_TASK_ID;
      public:
        DeferPerformTraversalArgs(PhysicalAnalysis *ana, EquivalenceSet *set,
         const FieldMask &mask, RtUserEvent done, bool already_deferred = true);
      public:
        PhysicalAnalysis *const analysis;
        EquivalenceSet *const set;
        FieldMask *const mask;
        const RtUserEvent applied_event;
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
        const RtUserEvent applied_event;
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
        const RtUserEvent applied_event;
        const RtUserEvent done_event;
      };
      struct DeferPerformOutputArgs : 
        public LgTaskArgs<DeferPerformOutputArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_OUTPUT_TASK_ID;
      public:
        DeferPerformOutputArgs(PhysicalAnalysis *ana, 
                               const PhysicalTraceInfo &trace_info);
      public:
        PhysicalAnalysis *const analysis;
        const PhysicalTraceInfo *trace_info;
        const RtUserEvent applied_event;
        const ApUserEvent effects_event;
      };
    public:
      // Local physical analysis
      PhysicalAnalysis(Runtime *rt, Operation *op, unsigned index,
                       IndexSpaceExpression *expr, bool on_heap,
                       CollectiveMapping *mapping = NULL);
      // Remote physical analysis
      PhysicalAnalysis(Runtime *rt, AddressSpaceID source, AddressSpaceID prev,
                       Operation *op, unsigned index, 
                       IndexSpaceExpression *expr, bool on_heap,
                       CollectiveMapping *mapping = NULL);
      PhysicalAnalysis(const PhysicalAnalysis &rhs);
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
    public:
      void traverse(EquivalenceSet *set, const FieldMask &mask, 
                    std::set<RtEvent> &deferral_events,
                    std::set<RtEvent> &applied_events,
                    RtEvent precondition = RtEvent::NO_RT_EVENT,
                    const bool already_deferred = false);
      void defer_traversal(RtEvent precondition, EquivalenceSet *set,
              const FieldMask &mask, std::set<RtEvent> &deferral_events, 
              std::set<RtEvent> &applied_events,
              RtUserEvent deferral_event = RtUserEvent::NO_RT_USER_EVENT,
              const bool already_deferred = true);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
      static void handle_deferred_remote(const void *args);
      static void handle_deferred_update(const void *args);
      static void handle_deferred_output(const void *args);
    public:
      const AddressSpaceID previous;
      const AddressSpaceID original_source;
      Runtime *const runtime;
      IndexSpaceExpression *const analysis_expr;
      CollectiveMapping *const collective_mapping;
      Operation *const op;
      const unsigned index;
      const bool owns_op;
      const bool on_heap;
    protected:
      LegionMap<AddressSpaceID,
                FieldMaskSet<EquivalenceSet> >::aligned remote_sets; 
      FieldMaskSet<LogicalView> *recorded_instances;
      bool restricted;
    private:
      // This tracks whether this analysis is being used 
      // for parallel traversals or not
      bool parallel_traversals;
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
      ValidInstAnalysis(const ValidInstAnalysis &rhs);
      virtual ~ValidInstAnalysis(void);
    public:
      ValidInstAnalysis& operator=(const ValidInstAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
      InvalidInstAnalysis(const InvalidInstAnalysis &rhs);
      virtual ~InvalidInstAnalysis(void);
    public:
      InvalidInstAnalysis& operator=(const InvalidInstAnalysis &rhs);
    public:
      inline bool has_invalid(void) const
      { return ((recorded_instances != NULL) && !recorded_instances->empty()); }
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
      AntivalidInstAnalysis(const AntivalidInstAnalysis &rhs);
      virtual ~AntivalidInstAnalysis(void);
    public:
      AntivalidInstAnalysis& operator=(const AntivalidInstAnalysis &rhs);
    public:
      inline bool has_antivalid(void) const
      { return ((recorded_instances != NULL) && !recorded_instances->empty()); }
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
    class UpdateAnalysis : public PhysicalAnalysis,
                           public LegionHeapify<UpdateAnalysis> {
    public:
      UpdateAnalysis(Runtime *rt, Operation *op, unsigned index,
                     const RegionRequirement &req,
                     RegionNode *node, const InstanceSet &target_instances,
                     std::vector<InstanceView*> &target_views,
                     std::vector<InstanceView*> &source_views,
                     const PhysicalTraceInfo &trace_info,
                     const ApEvent precondition, const ApEvent term_event,
                     const bool track_effects, const bool check_initialized,
                     const bool record_valid, const bool skip_output);
      UpdateAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                     Operation *op, unsigned index,
                     const RegionUsage &usage, RegionNode *node, 
                     InstanceSet &target_instances,
                     std::vector<InstanceView*> &target_views,
                     std::vector<InstanceView*> &source_views,
                     const PhysicalTraceInfo &trace_info,
                     const RtEvent user_registered,
                     const ApEvent precondition, const ApEvent term_event,
                     const bool track_effects, const bool check_initialized,
                     const bool record_valid, const bool skip_output);
      UpdateAnalysis(const UpdateAnalysis &rhs);
      virtual ~UpdateAnalysis(void);
    public:
      UpdateAnalysis& operator=(const UpdateAnalysis &rhs);
    public:
      bool has_output_updates(void) const 
        { return (output_aggregator != NULL); }
      void record_uninitialized(const FieldMask &uninit,
                                std::set<RtEvent> &applied_events);
      virtual void perform_traversal(EquivalenceSet *set,
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
      static void handle_remote_updates(Deserializer &derez, Runtime *rt,
                                        AddressSpaceID previous);
    public:
      // TODO: make this const again after we update the runtime to 
      // support collective views so that we don't need to modify 
      // this field in ReplMapOp::trigger_mapping
      /*const*/ RegionUsage usage;
      RegionNode *const node;
      const InstanceSet target_instances;
      const std::vector<InstanceView*> target_views;
      const std::vector<InstanceView*> source_views;
      const PhysicalTraceInfo trace_info;
      const ApEvent precondition;
      const ApEvent term_event;
      const bool track_effects;
      const bool check_initialized;
      const bool record_valid;
      const bool skip_output;
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
      std::set<ApEvent> effects_events;
    };

    /**
     * \class AcquireAnalysis
     * For performing acquires on equivalence set trees
     */
    class AcquireAnalysis : public PhysicalAnalysis,
                            public LegionHeapify<AcquireAnalysis> {
    public: 
      AcquireAnalysis(Runtime *rt, Operation *op, unsigned index,
                      IndexSpaceExpression *expr);
      AcquireAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                      Operation *op, unsigned index, IndexSpaceExpression *expr,
                      AcquireAnalysis *target); 
      AcquireAnalysis(const AcquireAnalysis &rhs);
      virtual ~AcquireAnalysis(void);
    public:
      AcquireAnalysis& operator=(const AcquireAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
    class ReleaseAnalysis : public PhysicalAnalysis,
                            public LegionHeapify<ReleaseAnalysis> {
    public:
      ReleaseAnalysis(Runtime *rt, Operation *op, unsigned index,
                      ApEvent precondition, IndexSpaceExpression *expr,
                      std::vector<InstanceView*> &source_views,
                      const PhysicalTraceInfo &trace_info);
      ReleaseAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                      Operation *op, unsigned index, IndexSpaceExpression *expr,
                      ApEvent precondition, ReleaseAnalysis *target, 
                      std::vector<InstanceView*> &source_views,
                      const PhysicalTraceInfo &info);
      ReleaseAnalysis(const ReleaseAnalysis &rhs);
      virtual ~ReleaseAnalysis(void);
    public:
      ReleaseAnalysis& operator=(const ReleaseAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
      const std::vector<InstanceView*> source_views;
      const PhysicalTraceInfo trace_info;
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
                         const std::vector<InstanceView*> &target_views,
                         const std::vector<InstanceView*> &source_views,
                         const ApEvent precondition,
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
                         const InstanceSet &target_instances,
                         const std::vector<InstanceView*> &target_views,
                         const std::vector<InstanceView*> &source_views,
                         const ApEvent precondition,
                         const PredEvent pred_guard, const ReductionOpID redop,
                         const std::vector<unsigned> &src_indexes,
                         const std::vector<unsigned> &dst_indexes,
                         const PhysicalTraceInfo &trace_info,
                         const bool perfect);
      CopyAcrossAnalysis(const CopyAcrossAnalysis &rhs);
      virtual ~CopyAcrossAnalysis(void);
    public:
      CopyAcrossAnalysis& operator=(const CopyAcrossAnalysis &rhs);
    public:
      bool has_across_updates(void) const 
        { return (across_aggregator != NULL); }
      void record_uninitialized(const FieldMask &uninit,
                                std::set<RtEvent> &applied_events);
      CopyFillAggregator* get_across_aggregator(void);
      virtual void perform_traversal(EquivalenceSet *set,
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
                            const InstanceSet &dst_instances,
                            const std::vector<unsigned> &src_indexes,
                            const std::vector<unsigned> &dst_indexes);
    public:
      const FieldMask src_mask;
      const FieldMask dst_mask;
      const unsigned src_index;
      const unsigned dst_index;
      const RegionUsage src_usage;
      const RegionUsage dst_usage;
      const LogicalRegion src_region;
      const LogicalRegion dst_region;
      const InstanceSet target_instances;
      const std::vector<InstanceView*> target_views;
      const std::vector<InstanceView*> source_views;
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
      std::set<ApEvent> copy_events;
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
                        const ApEvent precondition,
                        const RtEvent guard_event = RtEvent::NO_RT_EVENT,
                        const PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
                        const bool track_effects = false,
                        const bool add_restriction = false);
      // Also local but with a full set of views
      OverwriteAnalysis(Runtime *rt, Operation *op, unsigned index,
                        const RegionUsage &usage, IndexSpaceExpression *expr,
                        const FieldMaskSet<LogicalView> &views,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const RtEvent guard_event = RtEvent::NO_RT_EVENT,
                        const PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
                        const bool track_effects = false,
                        const bool add_restriction = false);
      OverwriteAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                        Operation *op, unsigned index,
                        IndexSpaceExpression *expr, const RegionUsage &usage, 
                        FieldMaskSet<LogicalView> &views,
                        FieldMaskSet<ReductionView> &reduction_views,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const RtEvent guard_event,
                        const PredEvent pred_guard,
                        const bool track_effects,
                        const bool add_restriction);
      OverwriteAnalysis(const OverwriteAnalysis &rhs);
      virtual ~OverwriteAnalysis(void);
    public:
      OverwriteAnalysis& operator=(const OverwriteAnalysis &rhs);
    public:
      bool has_output_updates(void) const 
        { return (output_aggregator != NULL); }
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
      static void handle_remote_overwrites(Deserializer &derez, Runtime *rt,
                                           AddressSpaceID previous); 
    public:
      const RegionUsage usage;
      FieldMaskSet<LogicalView> views;
      FieldMaskSet<ReductionView> reduction_views;
      const PhysicalTraceInfo trace_info;
      const ApEvent precondition;
      const RtEvent guard_event;
      const PredEvent pred_guard;
      const bool track_effects;
      const bool add_restriction;
    public:
      // Can only safely be accessed when analysis is locked
      CopyFillAggregator *output_aggregator;
      std::set<ApEvent> effects_events;
    };

    /**
     * \class FilterAnalysis
     * For performing filter traversals on equivalence set trees
     */
    class FilterAnalysis : public PhysicalAnalysis,
                           public LegionHeapify<FilterAnalysis> {
    public:
      FilterAnalysis(Runtime *rt, Operation *op, unsigned index,
                     IndexSpaceExpression *expr, InstanceView *inst_view,
                     LogicalView *registration_view,
                     const bool remove_restriction = false);
      FilterAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                     Operation *op, unsigned index, IndexSpaceExpression *expr,
                     InstanceView *inst_view, LogicalView *registration_view,
                     const bool remove_restriction);
      FilterAnalysis(const FilterAnalysis &rhs);
      virtual ~FilterAnalysis(void);
    public:
      FilterAnalysis& operator=(const FilterAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
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
      InstanceView *const inst_view;
      LogicalView *const registration_view;
      const bool remove_restriction;
    };

    /**
     * \class EqSetTracker
     * This is an abstract class that provides an interface for
     * recording the equivalence sets that result from ray tracing
     * an equivalence set tree for a given index space expression.
     */
    class EqSetTracker {
    public:
      virtual ~EqSetTracker(void) { }
    public:
      virtual void add_tracker_reference(unsigned cnt = 1) = 0;
      virtual bool remove_tracker_reference(unsigned cnt = 1) = 0;
    public:
      virtual void record_equivalence_set(EquivalenceSet *set,
                                          const FieldMask &mask) = 0;
      virtual void record_pending_equivalence_set(EquivalenceSet *set,
                                          const FieldMask &mask) = 0;
      virtual void remove_equivalence_set(EquivalenceSet *set,
                                          const FieldMask &mask) = 0;
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
#ifdef NEWEQ
    class EquivalenceSet : public DistributedCollectable,
                           public LegionHeapify<EquivalenceSet> {
    public:
      static const AllocationType alloc_type = EQUIVALENCE_SET_ALLOC;
    public:
      struct RefinementTaskArgs : public LgTaskArgs<RefinementTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REFINEMENT_TASK_ID;
      public:
        RefinementTaskArgs(EquivalenceSet *t)
          : LgTaskArgs<RefinementTaskArgs>(implicit_provenance), 
            target(t) { }
      public:
        EquivalenceSet *const target;
      };
      struct RemoteRefTaskArgs : public LgTaskArgs<RemoteRefTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_REF_TASK_ID;
      public:
        RemoteRefTaskArgs(DistributedID id, RtUserEvent done, bool add,
                          std::map<LogicalView*,unsigned> *r)
          : LgTaskArgs<RemoteRefTaskArgs>(implicit_provenance), 
            did(id), done_event(done), add_references(add), refs(r) { }
      public:
        const DistributedID did;
        const RtUserEvent done_event;
        const bool add_references;
        std::map<LogicalView*,unsigned> *refs;
      };
      struct DeferRayTraceArgs : public LgTaskArgs<DeferRayTraceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_RAY_TRACE_TASK_ID;
      public:
        DeferRayTraceArgs(EquivalenceSet *s, RayTracer *t,
                          IndexSpaceExpression *e, IndexSpace h,
                          AddressSpaceID o, RtUserEvent d,
                          RtUserEvent def, const FieldMask &m,
                          // This is to tell whether the ray tracing
                          // must be done purely symbolically.
                          // Must be true only for index spaces of
                          // output regions.
                          bool symbolic=false,
                          // These are just for the case where the
                          // request comes from a remote node and
                          // we're waiting for the expression to load
                          bool is_local=true, bool is_expr_s = false,
                          IndexSpace expr_h = IndexSpace::NO_SPACE,
                          IndexSpaceExprID expr_i = 0);
      public:
        EquivalenceSet *const set;
        RayTracer *const target;
        IndexSpaceExpression *const expr;
        const IndexSpace handle;
        const AddressSpaceID origin;
        const RtUserEvent done;
        const RtUserEvent deferral;
        FieldMask *const ray_mask;
        bool symbolic;
        const IndexSpace expr_handle;
        const IndexSpaceExprID expr_id;
        const bool is_local;
        const bool is_expr_space;
      };
      struct DeferRayTraceFinishArgs : 
        public LgTaskArgs<DeferRayTraceFinishArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_RAY_TRACE_FINISH_TASK_ID;
      public:
        DeferRayTraceFinishArgs(RayTracer *t, AddressSpaceID src,
            FieldMaskSet<EquivalenceSet> *to_tv,
            std::map<EquivalenceSet*,IndexSpaceExpression*> *exs,
            const size_t v, const IndexSpaceExprID e,
            const IndexSpace h, RtUserEvent d, bool sym)
          : LgTaskArgs<DeferRayTraceFinishArgs>(implicit_provenance),
            target(t), source(src), to_traverse(to_tv), exprs(exs), 
            volume(v), expr_id(e), handle(h), done(d), symbolic(sym) { }
      public:
        RayTracer *const target;
        const AddressSpaceID source;
        FieldMaskSet<EquivalenceSet> *const to_traverse;
        std::map<EquivalenceSet*,IndexSpaceExpression*> *const exprs;
        const size_t volume;
        const IndexSpaceExprID expr_id;
        const IndexSpace handle;
        const RtUserEvent done;
        const bool symbolic;
      };
      struct DeferSubsetRequestArgs : 
        public LgTaskArgs<DeferSubsetRequestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_SUBSET_REQUEST_TASK_ID;
      public:
        DeferSubsetRequestArgs(EquivalenceSet *s, 
                               AddressSpaceID src, RtUserEvent d)
          : LgTaskArgs<DeferSubsetRequestArgs>(implicit_provenance),
            set(s), source(src), deferral(d) { }
      public:
        EquivalenceSet *const set;
        const AddressSpaceID source;
        const RtUserEvent deferral;
      };
      struct DeferMakeOwnerArgs : public LgTaskArgs<DeferMakeOwnerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAKE_OWNER_TASK_ID;
      public:
        DeferMakeOwnerArgs(EquivalenceSet *s,
                           FieldMaskSet<EquivalenceSet> *sub, RtUserEvent d)
          : LgTaskArgs<DeferMakeOwnerArgs>(implicit_provenance), set(s),
            new_subsets(sub), done(d) { }
      public:
        EquivalenceSet *const set;
        FieldMaskSet<EquivalenceSet> *const new_subsets;
        const RtUserEvent done;
      };
      struct DeferMergeOrForwardArgs : 
        public LgTaskArgs<DeferMergeOrForwardArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MERGE_OR_FORWARD_TASK_ID;
      public:
        DeferMergeOrForwardArgs(EquivalenceSet *s, bool init,
            FieldMaskSet<LogicalView> *v,
            std::map<unsigned,std::vector<ReductionView*> > *r,
            FieldMaskSet<InstanceView> *t, 
            LegionMap<VersionID,FieldMask>::aligned *u, RtUserEvent d)
          : LgTaskArgs<DeferMergeOrForwardArgs>(implicit_provenance),
            set(s), views(v), reductions(r), restricted(t), 
            versions(u), done(d), initial(init) { }
      public:
        EquivalenceSet *const set;
        FieldMaskSet<LogicalView> *const views;
        std::map<unsigned,std::vector<ReductionView*> > *const reductions;
        FieldMaskSet<InstanceView> *const restricted;
        LegionMap<VersionID,FieldMask>::aligned *const versions;
        const RtUserEvent done;
        const bool initial;
      };
      struct DeferResponseArgs : public LgTaskArgs<DeferResponseArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_EQ_RESPONSE_TASK_ID;
      public:
        DeferResponseArgs(DistributedID id, AddressSpaceID src, 
                          AddressSpaceID log, IndexSpaceExpression *ex, 
                          bool local, bool is_space, IndexSpace expr_h, 
                          IndexSpaceExprID xid, IndexSpace h);
      public:
        const DistributedID did;
        const AddressSpaceID source;
        const AddressSpaceID logical_owner;
        IndexSpaceExpression *const expr;
        const bool is_local;
        const bool is_index_space;
        const IndexSpace expr_handle;
        const IndexSpaceExprID expr_id;
        const IndexSpace handle;
      };
      struct DeferRemoveRefArgs : public LgTaskArgs<DeferRemoveRefArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_REMOVE_EQ_REF_TASK_ID;
      public:
        DeferRemoveRefArgs(std::vector<IndexSpaceExpression*> *refs)
          : LgTaskArgs<DeferRemoveRefArgs>(implicit_provenance),
            references(refs) { }
      public:
        std::vector<IndexSpaceExpression*> *const references;
      };
    protected:
      enum EqState {
        // Owner starts in the mapping state, goes to pending refinement
        // once there are any refinements to be done which will wait for
        // all mappings to finish and then goes to refined once any 
        // refinements have been done
        MAPPING_STATE, // subsets is stable and no refinements being performed
        REFINING_STATE, // running the refinement task
        // Remote copies start in the invalid state, go to pending valid
        // while waiting for a lease on the current subsets, valid once they 
        // get a lease, pending invalid once they get an invalid notification
        // but have outsanding mappings, followed by invalid
        INVALID_STATE,
        PENDING_VALID_STATE,
        VALID_STATE,
      };
    protected:
      struct DisjointPartitionRefinement {
      public:
        DisjointPartitionRefinement(EquivalenceSet *owner, IndexPartNode *p,
                                    std::set<RtEvent> &applied_events,
                                    bool symbolic=false);
        DisjointPartitionRefinement(const DisjointPartitionRefinement &rhs,
                                    std::set<RtEvent> &applied_events);
        ~DisjointPartitionRefinement(void);
      public:
        inline const std::map<IndexSpaceNode*,EquivalenceSet*>& 
          get_children(void) const { return children; }
        inline bool is_refined(void) const 
          { return (total_child_volume == partition_volume); } 
        inline size_t get_volume(void) const { return partition_volume; }
      public:
        void add_child(IndexSpaceNode *node, EquivalenceSet *child);
        EquivalenceSet* find_child(IndexSpaceNode *node) const;
      public:
        const DistributedID owner_did;
        IndexPartNode *const partition;
      private:
        std::map<IndexSpaceNode*,EquivalenceSet*> children;
        size_t total_child_volume;
        const size_t partition_volume;
        const bool symbolic;
      };
    public:
      EquivalenceSet(Runtime *rt, DistributedID did,
                     AddressSpaceID owner_space,
                     AddressSpaceID logical_owner,
                     IndexSpaceExpression *expr, 
                     IndexSpaceNode *index_space_node,
                     bool register_now);
      EquivalenceSet(const EquivalenceSet &rhs);
      virtual ~EquivalenceSet(void);
    public:
      EquivalenceSet& operator=(const EquivalenceSet &rhs);
    public:
      inline bool has_refinements(const FieldMask &mask) const
        {
          AutoLock eq(eq_lock,1,false/*exclusive*/);
          return is_refined(mask);
        }
    public:
      // Must be called while holding the lock
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner_space); }
    protected:
      // Must be called while holding the lock
      inline bool is_refined(const FieldMask &mask) const
        { 
          return (!subsets.empty() && !(subsets.get_valid_mask() * mask)) ||
                  !(refining_fields * mask);
        }
    protected:
      inline void increment_pending_analyses(void)
        { pending_analyses++; }
      inline void decrement_pending_analyses(void)
        {
#ifdef DEBUG_LEGION
          assert(pending_analyses > 0);
#endif
          if ((--pending_analyses == 0) && !is_logical_owner() &&
              waiting_event.exists())
            // Signal to the migration task that it is safe to unpack
            trigger_pending_analysis_event();
        }
      // Need a separte function because Runtime::trigger_event is not included
      void trigger_pending_analysis_event(void);
    public:
      // From distributed collectable
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      AddressSpaceID clone_from(const EquivalenceSet *parent, 
                                const FieldMask &clone_mask);
      void remove_update_guard(CopyFillGuard *guard);
      void refresh_refinement(RayTracer *target, const FieldMask &mask,
                              RtUserEvent refresh_done);
      void ray_trace_equivalence_sets(RayTracer *target,
                                      IndexSpaceExpression *expr, 
                                      FieldMask ray_mask,
                                      IndexSpace handle,
                                      AddressSpaceID source,
                                      RtUserEvent ready,
                                      bool symbolic=false,
                                      RtUserEvent deferral_event = 
                                        RtUserEvent::NO_RT_USER_EVENT); 
      void record_subset(EquivalenceSet *set, const FieldMask &mask);
    public:
      // Analysis methods
      inline bool has_restrictions(const FieldMask &mask) const
        { return !(mask * restricted_fields); }
      FieldMask is_restricted(InstanceView *view);
      void initialize_set(const RegionUsage &usage,
                          const FieldMask &user_mask,
                          const bool restricted,
                          const InstanceSet &sources,
            const std::vector<InstanceView*> &corresponding,
                          std::set<RtEvent> &applied_events);
      void find_valid_instances(ValidInstAnalysis &analysis,
                                FieldMask user_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool cached_set,
                                const bool already_deferred = false);
      void find_invalid_instances(InvalidInstAnalysis &analysis,
                                FieldMask user_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                const bool cached_set,
                                const bool already_deferred = false);
      void update_set(UpdateAnalysis &analysis, FieldMask user_mask,
                      std::set<RtEvent> &deferral_events,
                      std::set<RtEvent> &applied_events,
                      FieldMask *stale_mask = NULL,
                      const bool cached_set = true,
                      const bool already_deferred = false);
    protected:
      void update_set_internal(CopyFillAggregator *&input_aggregator,
                               const RtEvent guard_event,
                               Operation *op, const unsigned index,
                               const RegionUsage &usage,
                               const FieldMask &user_mask,
                               const InstanceSet &target_instances,
                               const std::vector<InstanceView*> &target_views,
                               std::set<RtEvent> &applied_events,
                               const bool record_valid);
      void check_for_migration(PhysicalAnalysis &analysis,
                               std::set<RtEvent> &applied_events);
    public:
      void acquire_restrictions(AcquireAnalysis &analysis, 
                                FieldMask acquire_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                FieldMask *stale_mask = NULL,
                                const bool cached_set = true,
                                const bool already_deferred = false);
      void release_restrictions(ReleaseAnalysis &analysis,
                                FieldMask release_mask,
                                std::set<RtEvent> &deferral_events,
                                std::set<RtEvent> &applied_events,
                                FieldMask *stale_mask = NULL,
                                const bool cached_set = true,
                                const bool already_deferred = false);
      void issue_across_copies(CopyAcrossAnalysis &analysis,
                               FieldMask src_mask, 
                               IndexSpaceExpression *overlap,
                               std::set<RtEvent> &deferral_events,
                               std::set<RtEvent> &applied_events);
      void overwrite_set(OverwriteAnalysis &analysis, FieldMask mask,
                         std::set<RtEvent> &deferral_events,
                         std::set<RtEvent> &applied_events,
                         FieldMask *stale_mask = NULL,
                         const bool cached_set = true,
                         const bool already_deferred = false);
      void filter_set(FilterAnalysis &analysis, FieldMask mask,
                      std::set<RtEvent> &deferral_events,
                      std::set<RtEvent> &applied_events,
                      FieldMask *stale_mask = NULL,
                      const bool cached_set = true,
                      const bool already_deferred = false);
    protected:
      void request_remote_subsets(std::set<RtEvent> &applied_events);
      // Help for analysis, all must be called while holding the lock
      void record_instances(const FieldMask &record_mask, 
                            const InstanceSet &target_instances,
                            const std::vector<InstanceView*> &target_views,
                                  ReferenceMutator &mutator);
      void issue_update_copies_and_fills(CopyFillAggregator *&aggregator,
                                         const RtEvent guard_event,
                                         Operation *op, const unsigned index,
                                         const bool track_events,
                                         FieldMask update_mask,
                                         const InstanceSet &target_instances,
                         const std::vector<InstanceView*> &target_views,
                                         IndexSpaceExpression *expr,
                                         const bool skip_check = false,
                                         const int dst_index = -1) const;
      void filter_valid_instances(const FieldMask &filter_mask);
      void filter_reduction_instances(const FieldMask &filter_mask);
      void apply_reductions(const FieldMask &reduce_mask, 
                            CopyFillAggregator *&aggregator,
                            RtEvent guard_event, Operation *op,
                            const unsigned index, const bool track_events);
      void copy_out(const FieldMask &restricted_mask,
                    const InstanceSet &src_instances,
                    const std::vector<InstanceView*> &src_views,
                    Operation *op, const unsigned index,
                          CopyFillAggregator *&aggregator) const;
      // For the case with a single instance and a logical view
      void copy_out(const FieldMask &restricted_mask, LogicalView *src_view,
                    Operation *op, const unsigned index,
                          CopyFillAggregator *&aggregator) const;
      void advance_version_numbers(FieldMask advance_mask);
    protected:
      void perform_refinements(void);
      void check_for_unrefined_remainder(AutoLock &eq,
                                         const FieldMask &mask,
                                         AddressSpaceID source);
      void finalize_disjoint_refinement(DisjointPartitionRefinement *dis,
                                        const FieldMask &finalize_mask);
      void filter_unrefined_remainders(FieldMask &to_filter,
                                       IndexSpaceExpression *expr);
      void send_equivalence_set(AddressSpaceID target);
      // Must be called while holding the lock in exlcusive mode
      EquivalenceSet* add_pending_refinement(IndexSpaceExpression *expr,
                                             const FieldMask &mask,
                                             IndexSpaceNode *node,
                                             AddressSpaceID source);
      void process_subset_request(AddressSpaceID source,
                                  RtUserEvent deferral_event);
      void process_subset_response(Deserializer &derez);
      void process_subset_update(Deserializer &derez);
      void pack_state(Serializer &rez, const FieldMask &mask) const;
      void unpack_state(Deserializer &derez); 
      void merge_or_forward(const RtUserEvent done_event,
          bool initial_refinement, const FieldMaskSet<LogicalView> &new_views,
          const std::map<unsigned,std::vector<ReductionView*> > &new_reductions,
          const FieldMaskSet<InstanceView> &new_restrictions,
          const LegionMap<VersionID,FieldMask>::aligned &new_versions);
      void pack_migration(Serializer &rez, RtEvent done_migration);
      void unpack_migration(Deserializer &derez, AddressSpaceID source,
                            RtUserEvent done_event);
      bool make_owner(FieldMaskSet<EquivalenceSet> *new_subsets,
                      RtUserEvent done_event, bool need_lock);
      void update_owner(const AddressSpaceID new_logical_owner);
      void defer_traversal(AutoTryLock &eq, PhysicalAnalysis &analysis,
                           const FieldMask &mask,
                           std::set<RtEvent> &deferral_events,
                           std::set<RtEvent> &applied_events,
                           const bool already_deferred,
                           const bool cached_set);
      inline RtEvent chain_deferral_events(RtUserEvent deferral_event)
      {
        volatile Realm::Event::id_t *ptr = &next_deferral_precondition.id;
        RtEvent continuation_pre;
        do {
          continuation_pre.id = *ptr;
        } while (!__sync_bool_compare_and_swap(ptr,
                  continuation_pre.id, deferral_event.id));
        return continuation_pre;
      }
    public:
      static void handle_refinement(const void *args);
      static void handle_remote_references(const void *args);
      static void handle_ray_trace(const void *args, Runtime *runtime);
      static void handle_ray_trace_finish(const void *args);
      static void handle_subset_request(const void *args);
      static void handle_make_owner(const void *args);
      static void handle_merge_or_forward(const void *args);
      static void handle_deferred_response(const void *args, Runtime *runtime);
      static void handle_deferred_remove_refs(const void *args);
      static void handle_equivalence_set_request(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_equivalence_set_response(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_subset_request(Deserializer &derez, Runtime *runtime);
      static void handle_subset_response(Deserializer &derez, Runtime *runtime);
      static void handle_subset_update(Deserializer &derez, Runtime *rt);
      static void handle_ray_trace_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_ray_trace_response(Deserializer &derez, Runtime *rt);
      static void handle_migration(Deserializer &derez, 
                                   Runtime *runtime, AddressSpaceID source);
      static void handle_owner_update(Deserializer &derez, Runtime *rt);
      static void handle_remote_refinement(Deserializer &derez, Runtime *rt);
    public:
      IndexSpaceExpression *const set_expr;
      IndexSpaceNode *const index_space_node; // can be NULL
    protected:
      AddressSpaceID logical_owner_space;
      mutable LocalLock eq_lock;
    protected:
      // This is the actual physical state of the equivalence class
      FieldMaskSet<LogicalView>                           valid_instances;
      std::map<unsigned/*field idx*/,
               std::vector<ReductionView*> >              reduction_instances;
      FieldMask                                           reduction_fields;
      FieldMaskSet<InstanceView>                          restricted_instances;
      FieldMask                                           restricted_fields;
      // This is the current version number of the equivalence set
      // Each field should appear in exactly one mask
      LegionMap<VersionID,FieldMask>::aligned             version_numbers;
    protected:
      // Track the current state of this equivalence state
      EqState eq_state;
      // Fields that are being refined
      FieldMask refining_fields;
      // This tracks the most recent copy-fill aggregator for each field
      FieldMaskSet<CopyFillGuard> update_guards;
      // Keep track of the refinements that need to be done
      FieldMaskSet<EquivalenceSet> pending_refinements;
      // Record which remote eq sets are being refined for the first time
      std::set<EquivalenceSet*> remote_first_refinements;
      // Keep an event to track when the refinements are ready
      RtUserEvent transition_event;
      // An event to track when the refinement task is done on the owner
      // and when analyses are done on remote nodes for migration
      RtUserEvent waiting_event;
      // An event to order to deferral tasks
      volatile RtEvent next_deferral_precondition;
    protected:
      // If we have sub sets then we track those here
      // If this data structure is not empty, everything above is invalid
      // except for the remainder expression which is just waiting until
      // someone else decides that they need to access it
      FieldMaskSet<EquivalenceSet> subsets;
      std::map<IndexSpaceExpression*,EquivalenceSet*> *subset_exprs;
      // Set on the owner node for tracking the remote subset leases
      std::set<AddressSpaceID> remote_subsets;
      // Index space expression for unrefined remainder of our set_expr
      // This is only valid on the owner node
      FieldMaskSet<IndexSpaceExpression> unrefined_remainders;
      // For detecting when we are slicing a disjoint partition
      FieldMaskSet<DisjointPartitionRefinement> disjoint_partition_refinements;
    protected:
      // Uses these for determining when we should do migration
      // There is an implicit assumption here that equivalence sets
      // are only used be a small number of nodes that is less than
      // the samples per migration count, if it ever exceeds this 
      // then we'll issue a warning
      static const unsigned SAMPLES_PER_MIGRATION_TEST = 64;
      // How many total epochs we want to remember
      static const unsigned MIGRATION_EPOCHS = 2;
      std::vector<std::pair<AddressSpaceID,unsigned> > 
        user_samples[MIGRATION_EPOCHS];
      unsigned migration_index;
      unsigned sample_count;
      // Prevent migration while there are still analyses traversing the set
      unsigned pending_analyses;
    public:
      static const VersionID init_version = 1;
    };
#endif // NEWEQ

    class EquivalenceSet : public DistributedCollectable,
                           public LegionHeapify<EquivalenceSet> {
    public:
      template<bool INC>
      class ValidityFunctor {
      public:
        ValidityFunctor(EquivalenceSet *s, ReferenceMutator *m)
          : set(s), mutator(m) { }
      public:
        void apply(AddressSpaceID target)
          { if (INC) set->send_remote_gc_increment(target, mutator);
            else set->send_remote_gc_decrement(target, mutator); }
      public:
        EquivalenceSet *const set;
        ReferenceMutator *const mutator;
      };
      class InvalidateFunctor {
      public:
        InvalidateFunctor(DistributedID did, const FieldMask &mask,
                          std::set<RtEvent> &applied, AddressSpaceID origin,
                          const CollectiveMapping *mapping, Runtime *runtime);
      public:
        void apply(AddressSpaceID target);
      public:
        const DistributedID did;
        const FieldMask &mask;
        std::set<RtEvent> &applied;
        const AddressSpaceID origin;
        const CollectiveMapping *const invalidate_mapping;
        Runtime *const runtime;
      };
      class UpdateReplicatedFunctor {
      public:
        UpdateReplicatedFunctor(DistributedID did, const FieldMask &mask,
                                const CollectiveMapping *mapping, 
                                AddressSpaceID origin, AddressSpaceID to_skip,
                                Runtime *runtime, std::set<RtEvent> &applied);
      public:
        void apply(AddressSpaceID target);
      public:
        const DistributedID did;
        const FieldMask &mask;
        const AddressSpaceID origin;
        const AddressSpaceID to_skip;
        const CollectiveMapping *mapping;
        Runtime *const runtime;
        std::set<RtEvent> &applied;
      };
    public:
      struct PendingReplication {
      public:
        PendingReplication(CollectiveMapping *mapping, unsigned notifications);
        ~PendingReplication(void);
      public:
        CollectiveMapping *const mapping;
        const RtUserEvent ready_event;
        std::set<RtEvent> preconditions;
        unsigned remaining_notifications;
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
      struct DeferPendingReplicationArgs : 
        public LgTaskArgs<DeferPendingReplicationArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PENDING_REPLICATION_TASK_ID;
      public:
        DeferPendingReplicationArgs(EquivalenceSet *s, PendingReplication *p,
                                    const FieldMask &m);
      public:
        EquivalenceSet *const set;
        PendingReplication *const pending;
        FieldMask *const mask;
      };
      struct DeferApplyStateArgs : public LgTaskArgs<DeferApplyStateArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_APPLY_STATE_TASK_ID;
      public:
        DeferApplyStateArgs(EquivalenceSet *set, RtUserEvent done_event, 
                            const bool foward_to_owner);
      public:
        EquivalenceSet *const set;
        LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> >::aligned
          *const valid_updates;
        FieldMaskSet<IndexSpaceExpression> *const initialized_updates;
        std::map<unsigned,std::list<std::pair<ReductionView*,
          IndexSpaceExpression*> > > *const reduction_updates;
        LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> >::aligned
          *const restricted_updates;
        LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> >::aligned
          *const released_updates;
        FieldMaskSet<CopyFillGuard> *const read_only_updates;
        FieldMaskSet<CopyFillGuard> *const reduction_fill_updates;
        TraceViewSet *precondition_updates;
        TraceViewSet *anticondition_updates;
        TraceViewSet *postcondition_updates;
        const RtUserEvent done_event;
        const bool forward_to_owner;
      };
      struct DeferReleaseRefArgs : public LgTaskArgs<DeferReleaseRefArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_RELEASE_REF_TASK_ID;
      public:
        DeferReleaseRefArgs(DistributedID did);
      public:
        const DistributedID did;
        std::map<LogicalView*,unsigned> *const view_refs_to_remove;
        std::map<IndexSpaceExpression*,unsigned> *const expr_refs_to_remove;
      };
    public:
      EquivalenceSet(Runtime *rt, DistributedID did,
                     AddressSpaceID owner_space,
                     AddressSpaceID logical_owner,
                     RegionNode *region_node,
                     bool register_now, 
                     CollectiveMapping *mapping = NULL,
                     const FieldMask *replicated = NULL);
      EquivalenceSet(const EquivalenceSet &rhs);
      virtual ~EquivalenceSet(void);
    public:
      EquivalenceSet& operator=(const EquivalenceSet &rhs);
    public:
      // Must be called while holding the lock
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner_space); }
      inline bool has_replicated_fields(const FieldMask &mask) const
        { return !(mask - replicated_states.get_valid_mask()); }
      inline const FieldMask& get_replicated_fields(void) const
        { return replicated_states.get_valid_mask(); }
    public:
      // From distributed collectable
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      // Analysis methods
      void initialize_set(const RegionUsage &usage,
                          const FieldMask &user_mask,
                          const bool restricted,
                          const InstanceSet &sources,
            const std::vector<InstanceView*> &corresponding,
                          std::set<RtEvent> &applied_events);
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
    public:
      void initialize_collective_references(unsigned local_valid_refs);
      void remove_read_only_guard(CopyFillGuard *guard);
      void remove_reduction_fill_guard(CopyFillGuard *guard);
      void record_tracker(EqSetTracker *tracker, const FieldMask &mask);
      void remove_tracker(EqSetTracker *tracker, const FieldMask &mask);
      void invalidate_trackers(const FieldMask &mask,
                               std::set<RtEvent> &applied_events,
                               const AddressSpaceID origin_space,
                               const CollectiveMapping *collective_mapping);
      void clone_owner(EquivalenceSet *src, const FieldMask &clone_mask,
                      std::set<RtEvent> &applied_events, 
                      const bool invalidate_overlap = false);
      void clone_from(const AddressSpaceID target_space, EquivalenceSet *src,
                      const FieldMask &clone_mask,
                      const bool forward_to_owner,
                      std::set<RtEvent> &applied_events, 
                      const bool invalidate_overlap = false);
      RtEvent make_owner(AddressSpaceID owner, 
                         RtEvent precondition = RtEvent::NO_RT_EVENT);
      void update_tracing_valid_views(LogicalView *view,
                                      IndexSpaceExpression *expr,
                                      const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      const bool invalidates,
                                      std::set<RtEvent> &applied);
      void update_tracing_anti_views(LogicalView *view,
                                     IndexSpaceExpression *expr,
                                     const FieldMask &user_mask,
                                     std::set<RtEvent> &applied);
      RtEvent capture_trace_conditions(TraceConditionSet *target,
                                       AddressSpaceID target_space,
                                       IndexSpaceExpression *expr,
                                       const FieldMask &mask,
                                       RtUserEvent ready_event);
    protected:
      void defer_traversal(AutoTryLock &eq, PhysicalAnalysis &analysis,
                           const FieldMask &mask,
                           std::set<RtEvent> &deferral_events,
                           std::set<RtEvent> &applied_events,
                           const bool already_deferred);
      inline RtEvent chain_deferral_events(RtUserEvent deferral_event)
      {
        volatile Realm::Event::id_t *ptr = &next_deferral_precondition.id;
        RtEvent continuation_pre;
        do {
          continuation_pre.id = *ptr;
        } while (!__sync_bool_compare_and_swap(ptr,
                  continuation_pre.id, deferral_event.id));
        return continuation_pre;
      }
      bool is_remote_analysis(PhysicalAnalysis &analysis,
                              const FieldMask &mask, 
                              std::set<RtEvent> &deferral_events,
                              std::set<RtEvent> &applied_events,
                              const bool exclusive,
                              const bool immutable = false);
    protected:
      template<typename T>
      void check_for_uninitialized_data(T &analysis, IndexSpaceExpression *expr,
                                  const bool expr_cover, FieldMask uninit,
                                  std::set<RtEvent> &applied_events) const;
      void update_initialized_data(IndexSpaceExpression *expr, 
                            const bool expr_covers, const FieldMask &user_mask);
      template<typename T>
      void record_instances(IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &record_mask, 
                            const FieldMaskSet<T> &new_views,
                                  ReferenceMutator &mutator);
      template<typename T>
      void record_unrestricted_instances(IndexSpaceExpression *expr,
                            const bool expr_covers, FieldMask record_mask, 
                            const FieldMaskSet<T> &new_views,
                                  ReferenceMutator &mutator);
      bool record_partial_valid_instance(LogicalView *instance,
                                         IndexSpaceExpression *expr,
                                         FieldMask valid_mask,
                                         ReferenceMutator &mutator);
      void filter_valid_instances(IndexSpaceExpression *expr, 
                                  const bool expr_covers, 
                                  const FieldMask &filter_mask,
                                  ReferenceMutator &mutator,
           std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
           std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      void filter_unrestricted_instances(IndexSpaceExpression *expr,
                                         const bool expr_covers, 
                                         FieldMask filter_mask,
                                         ReferenceMutator &mutator);
      void filter_reduction_instances(IndexSpaceExpression *expr,
           const bool expr_covers, const FieldMask &filter_mask,
           std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
           std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      void update_set_internal(CopyFillAggregator *&input_aggregator,
                               const RtEvent guard_event,
                               Operation *op, const unsigned index,
                               const RegionUsage &usage,
                               IndexSpaceExpression *expr, 
                               const bool expr_covers,
                               const FieldMask &user_mask,
                               const FieldMaskSet<InstanceView> &target_insts,
                               const std::vector<InstanceView*> &source_insts,
                               const PhysicalTraceInfo &trace_info,
                               std::set<RtEvent> &applied_events,
                               const bool record_valid);
      void make_instances_valid(CopyFillAggregator *&aggregator,
                                const RtEvent guard_event,
                                Operation *op, const unsigned index,
                                const bool track_events,
                                IndexSpaceExpression *expr,
                                const bool expr_covers,
                                const FieldMask &update_mask,
                                const FieldMaskSet<InstanceView> &target_insts,
                                const std::vector<InstanceView*> &source_insts,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &applied_events,
                                const bool skip_check = false,
                                const int dst_index = -1,
                                const ReductionOpID redop = 0,
                                CopyAcrossHelper *across_helper = NULL);
      void issue_update_copies_and_fills(InstanceView *target,
                                const std::vector<InstanceView*> &source_views,
                                         CopyFillAggregator *&aggregator,
                                         const RtEvent guard_event,
                                         Operation *op, const unsigned index,
                                         const bool track_events,
                                         IndexSpaceExpression *expr,
                                         const bool expr_covers,
                                         FieldMask update_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &applied_events,
                                         const int dst_index,
                                         const ReductionOpID redop,
                                         CopyAcrossHelper *across_helper);
      void apply_reductions(const FieldMaskSet<InstanceView> &reduction_targets,
                            IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &reduction_mask, 
                            CopyFillAggregator *&aggregator,RtEvent guard_event,
                            Operation *op, const unsigned index, 
                            const bool track_events,
                            const PhysicalTraceInfo &trace_info,
                            std::set<RtEvent> &applied_events,
                            FieldMaskSet<IndexSpaceExpression> *applied_exprs,
                            CopyAcrossHelper *across_helper = NULL);
      template<typename T>
      void copy_out(IndexSpaceExpression *expr, const bool expr_covers,
                    const FieldMask &restricted_mask, 
                    const FieldMaskSet<T> &src_views,
                    Operation *op, const unsigned index,
                    const PhysicalTraceInfo &trace_info,
                    std::set<RtEvent> &applied_events,
                    CopyFillAggregator *&aggregator);
      void record_restriction(IndexSpaceExpression *expr, 
                              const bool expr_covers,
                              const FieldMask &restrict_mask,
                              InstanceView *restricted_view,
                              ReferenceMutator &mutator);
      void update_reductions(const unsigned fidx, ReferenceMutator &mutator,
          std::list<std::pair<ReductionView*,IndexSpaceExpression*> > &updates);
      void update_released(IndexSpaceExpression *expr, const bool expr_covers,
                FieldMaskSet<InstanceView> &updates, ReferenceMutator &mutator);
      void filter_initialized_data(IndexSpaceExpression *expr, 
          const bool expr_covers, const FieldMask &filter_mask, 
          std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL);
      void filter_restricted_instances(IndexSpaceExpression *expr, 
          const bool expr_covers, const FieldMask &filter_mask, 
          std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
          std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
      void filter_released_instances(IndexSpaceExpression *expr, 
          const bool expr_covers, const FieldMask &filter_mask, 
          std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
          std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL);
    protected:
      void send_equivalence_set(AddressSpaceID target);
      void check_for_migration(PhysicalAnalysis &analysis,
                               std::set<RtEvent> &applied_events);
      void update_owner(const AddressSpaceID new_logical_owner); 
      void broadcast_replicated_state_updates(const FieldMask &mask,
              CollectiveMapping *mapping, const AddressSpaceID origin,
              std::set<RtEvent> &applied_events, const bool need_lock = false,
              const bool perform_updates = true);
      void make_replicated_state(CollectiveMapping *mapping,
                                 FieldMask mask, const AddressSpaceID source,
                                 std::set<RtEvent> &deferral_events);
      void process_replication_request(const FieldMask &mask, 
                CollectiveMapping *mapping, PendingReplication *target, 
                const AddressSpaceID source, const RtEvent done_event);
      void process_replication_response(PendingReplication *target,
                const FieldMask &mask, RtEvent precondition,
                const FieldMask &update_mask, Deserializer &derez);
      void unpack_replicated_states(Deserializer &derez);
      void update_replicated_state(CollectiveMapping *mapping, 
                                   const FieldMask &mask);
      void finalize_pending_replication(PendingReplication *pending,
         const FieldMask &mask, const bool first, const bool need_lock = false);
    protected:
      void pack_state(Serializer &rez, const AddressSpaceID target,
            IndexSpaceExpression *expr, const bool expr_covers,
            const FieldMask &mask, const bool pack_guards);
      void unpack_state_and_apply(Deserializer &derez, 
          const AddressSpaceID source, const bool forward_to_owner,
          std::set<RtEvent> &ready_events);
      void invalidate_state(IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &mask, RtEvent remove_ref_event);
      void clone_to_local(EquivalenceSet *dst, FieldMask mask,
                          std::set<RtEvent> &applied_events,
                          const bool invalidate_overlap,
                          const bool forward_to_owner);
      void clone_to_remote(DistributedID target, AddressSpaceID target_space,
                    IndexSpaceNode *target_node, const FieldMask &mask,
                    RtUserEvent done_event, const bool invalidate_overlap,
                    const bool forward_to_owner);
      void find_overlap_updates(IndexSpaceExpression *overlap, 
            const bool overlap_covers, const FieldMask &mask, 
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> >::aligned &valid_updates,
            FieldMaskSet<IndexSpaceExpression> &initialized_updates,
            std::map<unsigned,std::list<std::pair<ReductionView*,
                IndexSpaceExpression*> > > &reduction_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::aligned &restricted_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::aligned &released_updates,
            FieldMaskSet<CopyFillGuard> *read_only_guard_updates,
            FieldMaskSet<CopyFillGuard> *reduction_fill_guard_updates,
            TraceViewSet *&precondition_updates,
            TraceViewSet *&anticondition_updates,
            TraceViewSet *&postcondition_updates) const;
      void apply_state(LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> >::aligned &valid_updates,
            FieldMaskSet<IndexSpaceExpression> &initialized_updates,
            std::map<unsigned,std::list<std::pair<ReductionView*,
                IndexSpaceExpression*> > > &reduction_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::aligned &restricted_updates,
            LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::aligned &released_updates,
            TraceViewSet *precondition_updates,
            TraceViewSet *anticondition_updates,
            TraceViewSet *postcondition_updates,
            FieldMaskSet<CopyFillGuard> *read_only_guard_updates,
            FieldMaskSet<CopyFillGuard> *reduction_fill_guard_updates,
            std::set<RtEvent> &applied_events,
            const bool needs_lock, const bool forward_to_owner);
      static void pack_updates(Serializer &rez, const AddressSpaceID target,
            const LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> >::aligned &valid_updates,
            const FieldMaskSet<IndexSpaceExpression> &initialized_updates,
            const std::map<unsigned,std::list<std::pair<ReductionView*,
                IndexSpaceExpression*> > > &reduction_updates,
            const LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::aligned &restricted_updates,
            const LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::aligned &released_updates,
            const FieldMaskSet<CopyFillGuard> *read_only_updates,
            const FieldMaskSet<CopyFillGuard> *reduction_fill_updates,
            const TraceViewSet *precondition_updates,
            const TraceViewSet *anticondition_updates,
            const TraceViewSet *postcondition_updates);
    public:
      static void handle_make_owner(const void *args);
      static void handle_pending_replication(const void *args);
      static void handle_apply_state(const void *args);
      static void handle_remove_refs(const void *args);
    public:
      static void handle_equivalence_set_request(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_equivalence_set_response(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_migration(Deserializer &derez, 
                                   Runtime *runtime, AddressSpaceID source);
      static void handle_owner_update(Deserializer &derez, Runtime *rt);
      static void handle_make_owner(Deserializer &derez, Runtime *rt);
      static void handle_invalidate_trackers(Deserializer &derez, Runtime *rt);
      static void handle_replication_request(Deserializer &derez, Runtime *rt);
      static void handle_replication_response(Deserializer &derez, Runtime *rt);
      static void handle_replication_update(Deserializer &derez, Runtime *rt);
      static void handle_clone_request(Deserializer &derez, Runtime *runtime);
      static void handle_clone_response(Deserializer &derez, Runtime *runtime);
      static void handle_capture_request(Deserializer &derez, Runtime *runtime,
                                         AddressSpaceID source);
      static void handle_capture_response(Deserializer &derez, Runtime *runtime,
                                          AddressSpaceID source);
    public:
      RegionNode *const region_node;
      IndexSpaceNode *const set_expr;
    protected:
      // This is the physical state of the equivalence set
      mutable LocalLock                                 eq_lock;
      FieldMaskSet<LogicalView>                         total_valid_instances;
      LegionMap<LogicalView*,
        FieldMaskSet<IndexSpaceExpression> >::aligned   partial_valid_instances;
      FieldMask                                         partial_valid_fields;
      // Expressions and fields that have valid data
      FieldMaskSet<IndexSpaceExpression>                initialized_data;
      // Reductions always need to be applied in order so keep them in order
      std::map<unsigned/*fidx*/,std::list<std::pair<
        ReductionView*,IndexSpaceExpression*> > >       reduction_instances;
      FieldMask                                         reduction_fields;
      // The list of expressions with the single instance for each
      // field that represents the restriction of that expression
      LegionMap<IndexSpaceExpression*,
        FieldMaskSet<InstanceView> >::aligned           restricted_instances;
      // Summary of any field that has a restriction
      FieldMask                                         restricted_fields;
      // List of instances that were restricted, but have been acquired
      LegionMap<IndexSpaceExpression*,
        FieldMaskSet<InstanceView> >::aligned           released_instances;
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
      volatile RtEvent                               next_deferral_precondition;
    protected:
      // This node is the node which contains the valid state data
      AddressSpaceID                                    logical_owner_space;
      // In control replicated cases, we allow equivalence sets to have
      // replicated state as long as their is a total sharding that updates
      // all the equivalence sets across the machine collectively.
      FieldMaskSet<CollectiveMapping>                   replicated_states;
      FieldMaskSet<PendingReplication>                  pending_states;
    protected:
      // Which EqSetTracker objects on this node are
      // tracking this equivalence set and need to be invalidated
      FieldMaskSet<EqSetTracker>                        recorded_trackers;
    protected:
      // Uses these for determining when we should do migration
      // There is an implicit assumption here that equivalence sets
      // are only used be a small number of nodes that is less than
      // the samples per migration count, if it ever exceeds this 
      // then we'll issue a warning
      static const unsigned SAMPLES_PER_MIGRATION_TEST = 64;
      // How many total epochs we want to remember
      static const unsigned MIGRATION_EPOCHS = 2;
      std::vector<std::pair<AddressSpaceID,unsigned> > 
        user_samples[MIGRATION_EPOCHS];
      unsigned migration_index;
      unsigned sample_count;
    protected:
      bool init_collective_refs;
#ifdef DEBUG_LEGION
    protected:
      // Make sure that each equivalence set only becomes active once
      bool active_once;
#endif
    };

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
      PendingEquivalenceSet(RegionNode *region_node, const FieldMask &mask);
      PendingEquivalenceSet(const PendingEquivalenceSet &rhs);
      ~PendingEquivalenceSet(void);
    public:
      PendingEquivalenceSet& operator=(const PendingEquivalenceSet &rhs);
    public:
      void record_previous(EquivalenceSet *set, const FieldMask &mask,
                           std::set<RtEvent> &applied_events);
      void record_all(VersionInfo &version_info, 
                      std::set<RtEvent> &applied_events);
    public:
      inline const FieldMask& get_valid_mask(void) const
        { return previous_sets.get_valid_mask(); }
      EquivalenceSet* compute_refinement(AddressSpaceID suggested_owner,
                      Runtime *runtime, std::set<RtEvent> &ready_events);
      bool finalize(const FieldMask &done_mask, bool &delete_now);
      static void handle_defer_finalize(const void *args);
    public:
      RegionNode *const region_node;
    protected:
      EquivalenceSet *new_set;
      RtEvent clone_event;
      FieldMaskSet<EquivalenceSet> previous_sets;
    };

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
      VersionManager(const VersionManager &manager);
      virtual ~VersionManager(void);
    public:
      VersionManager& operator=(const VersionManager &rhs);
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
                                       const UniqueID opid,
                                       const AddressSpaceID source,
                                       std::set<RtEvent> &ready);
      virtual void add_tracker_reference(unsigned cnt = 1);
      virtual bool remove_tracker_reference(unsigned cnt = 1);
      virtual void record_equivalence_set(EquivalenceSet *set, 
                                          const FieldMask &mask);
      virtual void record_pending_equivalence_set(EquivalenceSet *set, 
                                          const FieldMask &mask);
      virtual void remove_equivalence_set(EquivalenceSet *set,
                                          const FieldMask &mask);
      void finalize_equivalence_sets(RtUserEvent done_event);                           
      void finalize_manager(void);
    public:
      // Call these from region nodes
      void initialize_versioning_analysis(EquivalenceSet *set,
              const FieldMask &mask, std::set<RtEvent> &applied_events);
      void compute_equivalence_sets(const ContextID ctx,
                                    IndexSpaceExpression *expr,
                                    EqSetTracker *target, 
                                    const AddressSpaceID target_space,
                                    FieldMask mask, InnerContext *context,
                                    const UniqueID opid,
                                    const AddressSpaceID original_source,
                                    std::set<RtEvent> &ready_events,
                                    FieldMaskSet<PartitionNode> &children,
                                    FieldMask &parent_traversal,
                                    std::set<RtEvent> &deferral_events,
                                    const bool downward_only);
      void find_or_create_empty_equivalence_sets(EqSetTracker *target,
                                    const AddressSpaceID target_space,
                                    const FieldMask &mask,
                                    const AddressSpaceID source,
                                    std::set<RtEvent> &ready_events);
      static void handle_compute_equivalence_sets_response(
                      Deserializer &derez, Runtime *runtime);
      void record_refinement(EquivalenceSet *set, const FieldMask &mask,
                             FieldMask &parent_mask,
                             std::set<RtEvent> &applied_events);
    public:
      // Call these from partition nodes
      void compute_equivalence_sets(const FieldMask &mask,
                                    FieldMask &parent_traversal, 
                                    FieldMask &children_traversal) const;
      void propagate_refinement(const std::vector<RegionNode*> &children,
                                const FieldMask &child_mask, 
                                FieldMask &parent_mask,
                                std::set<RtEvent> &applied_events);
    public:
      // Call these from either type of region tree node
      void propagate_refinement(RegionTreeNode *child, 
                                const FieldMask &child_mask, 
                                FieldMask &parent_mask,
                                std::set<RtEvent> &applied_events);
      void invalidate_refinement(InnerContext *context,
                                 const FieldMask &mask, bool invalidate_self,
                                 FieldMaskSet<RegionTreeNode> &to_traverse,
                                 FieldMaskSet<EquivalenceSet> &to_untrack,
                                 std::vector<EquivalenceSet*> &to_release);
      void merge(VersionManager &src, std::set<RegionTreeNode*> &to_traverse,
                 FieldMaskSet<EquivalenceSet> &to_untrack);
      void swap(VersionManager &src, std::set<RegionTreeNode*> &to_traverse,
                FieldMaskSet<EquivalenceSet> &to_untrack);
      void pack_manager(Serializer &rez, const bool invalidate, 
                        std::map<LegionColor,RegionTreeNode*> &to_traverse,
                        FieldMaskSet<EquivalenceSet> &to_untrack,
                        std::vector<DistributedCollectable*> &to_remove);
      void unpack_manager(Deserializer &derez, AddressSpaceID source,
                          std::map<LegionColor,RegionTreeNode*> &to_traverse);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                                TreeStateLogger *logger);
    public:
      static void handle_finalize_eq_sets(const void *args);
    public:
      const ContextID ctx;
      RegionTreeNode *const node;
      Runtime *const runtime;
    protected:
      mutable LocalLock manager_lock;
    protected: 
      FieldMaskSet<EquivalenceSet> equivalence_sets;
      FieldMaskSet<EquivalenceSet> pending_equivalence_sets;
      LegionList<WaitingVersionInfo>::aligned waiting_infos;
      LegionMap<RtUserEvent,FieldMask>::aligned equivalence_sets_ready;
    protected:
      // The fields for which this node has disjoint complete information
      FieldMask disjoint_complete;
      // Track which disjoint and complete children we have from this
      // node for representing the refinement tree. Note that if this
      // context is control replicated this set might not be complete
      // for partition nodes, Some sub-region nodes might only exist
      // in contexts on remote shards.
      FieldMaskSet<RegionTreeNode> disjoint_complete_children;
      // We are sometimes lazy in filling in the equivalence sets for
      // disjoint-complete partitions from refinement ops so we can 
      // pick the logical owner from the first address space to attempt
      // to touch it. In that case we need a data structure to make
      // sure that there is only one call going out to the context
      // at a time for each field to make the equivalence sets.
      LegionMap<RtEvent,FieldMask>::aligned disjoint_complete_ready;
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
      void record_aliased_children(unsigned depth, const FieldMask &mask);
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
    public:
      const FieldMask* get_aliased_children(unsigned depth) const;
    protected:
      std::vector<LegionColor> path;
      LegionMap<unsigned/*depth*/,FieldMask>::aligned interfering_children;
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
