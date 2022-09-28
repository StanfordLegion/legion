/* Copyright 2022 Stanford University, NVIDIA Corporation
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
      inline bool has_version_info(void) const { return (owner != NULL); }
      inline const FieldMaskSet<EquivalenceSet>& get_equivalence_sets(void) 
        const { return equivalence_sets; }
      inline VersionManager* get_manager(void) const { return owner; }
    public:
      void record_equivalence_set(VersionManager *owner, 
                                  EquivalenceSet *set, 
                                  const FieldMask &set_mask);
      void clear(void);
    protected:
      VersionManager *owner;
      FieldMaskSet<EquivalenceSet> equivalence_sets;
    };

    /**
     * \struct LogicalTraceInfo
     * Information about tracing needed for logical
     * dependence analysis.
     */
    struct LogicalTraceInfo {
    public:
      LogicalTraceInfo(bool already_tr,
                       LegionTrace *tr,
                       unsigned idx,
                       const RegionRequirement &r);
    public:
      bool already_traced;
      LegionTrace *trace;
      unsigned req_idx;
      const RegionRequirement &req;
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
    public:
      virtual void record_get_term_event(Memoizable *memo) = 0;
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
      virtual void record_merge_events(ApEvent &lhs,
                         const std::vector<ApEvent>& rhs, Memoizable *memo) = 0;
    public:
      virtual void record_issue_copy(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
                           const std::vector<Reservation>& reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard) = 0;
      virtual void record_issue_across(Memoizable *memo, ApEvent &lhs,
                           ApEvent collective_precondition, 
                           ApEvent copy_precondition,
                           ApEvent src_indirect_precondition,
                           ApEvent dst_indirect_precondition,
                           CopyAcrossExecutor *executor) = 0;
      virtual void record_copy_views(ApEvent lhs, Memoizable *memo,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           bool src_indirect, bool dst_indirect,
                           std::set<RtEvent> &applied) = 0;
      virtual void record_indirect_views(ApEvent indirect_done,ApEvent all_done,
                           Memoizable *memo, unsigned index,
                           IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_views,
                           std::set<RtEvent> &applied, PrivilegeMode priv) = 0;
      virtual void record_issue_fill(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard) = 0;
      virtual void record_post_fill_view(FillView *view, 
                                         const FieldMask &mask) = 0;
      virtual void record_fill_views(ApEvent lhs, Memoizable *memo,
                           unsigned idx, IndexSpaceExpression *expr, 
                           const FieldMaskSet<FillView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied_events,
                           bool reduction_initialization) = 0;
    public:
      virtual void record_op_view(Memoizable *memo,
                          unsigned idx,
                          InstanceView *view,
                          const RegionUsage &usage,
                          const FieldMask &user_mask,
                          bool update_validity) = 0;
      virtual void record_set_op_sync_event(ApEvent &lhs, Memoizable *memo) = 0;
      virtual void record_mapper_output(const TraceLocalID &tlid,
                         const Mapper::MapTaskOutput &output,
                         const std::deque<InstanceSet> &physical_instances,
                         std::set<RtEvent> &applied_events) = 0;
      virtual void record_set_effects(Memoizable *memo, ApEvent &rhs) = 0;
      virtual void record_complete_replay(Memoizable *memo, ApEvent rhs) = 0;
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
        REMOTE_TRACE_RECORD_GET_TERM,
        REMOTE_TRACE_CREATE_USER_EVENT,
        REMOTE_TRACE_TRIGGER_EVENT,
        REMOTE_TRACE_MERGE_EVENTS,
        REMOTE_TRACE_ISSUE_COPY,
        REMOTE_TRACE_COPY_VIEWS,
        REMOTE_TRACE_INDIRECT_VIEWS,
        REMOTE_TRACE_ISSUE_FILL,
        REMOTE_TRACE_FILL_VIEWS,
        REMOTE_TRACE_RECORD_OP_VIEW,
        REMOTE_TRACE_SET_OP_SYNC,
        REMOTE_TRACE_SET_EFFECTS,
        REMOTE_TRACE_RECORD_MAPPER_OUTPUT,
        REMOTE_TRACE_COMPLETE_REPLAY,
        REMOTE_TRACE_ACQUIRE_RELEASE,
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
    public:
      virtual void record_get_term_event(Memoizable *memo);
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
      virtual void record_merge_events(ApEvent &lhs, 
                            const std::vector<ApEvent>& rhs, Memoizable *memo);
    public:
      virtual void record_issue_copy(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField>& src_fields,
                           const std::vector<CopySrcDstField>& dst_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
      virtual void record_issue_across(Memoizable *memo, ApEvent &lhs, 
                           ApEvent collective_precondition, 
                           ApEvent copy_precondition,
                           ApEvent src_indirect_precondition,
                           ApEvent dst_indirect_precondition,
                           CopyAcrossExecutor *executor);
      virtual void record_copy_views(ApEvent lhs, Memoizable *memo,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           bool src_indirect, bool dst_indirect,
                           std::set<RtEvent> &applied);
      virtual void record_indirect_views(ApEvent indirect_done,ApEvent all_done,
                           Memoizable *memo, unsigned index, 
                           IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_views,
                           std::set<RtEvent> &applied, PrivilegeMode priv);
      virtual void record_issue_fill(Memoizable *memo, ApEvent &lhs,
                           IndexSpaceExpression *expr,
                           const std::vector<CopySrcDstField> &fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
      virtual void record_post_fill_view(FillView *view, const FieldMask &mask);
      virtual void record_fill_views(ApEvent lhs, Memoizable *memo,
                           unsigned idx, IndexSpaceExpression *expr, 
                           const FieldMaskSet<FillView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied_events,
                           bool reduction_initialization);
    public:
      virtual void record_op_view(Memoizable *memo,
                          unsigned idx,
                          InstanceView *view,
                          const RegionUsage &usage,
                          const FieldMask &user_mask,
                          bool update_validity);
      virtual void record_set_op_sync_event(ApEvent &lhs, Memoizable *memo);
      virtual void record_mapper_output(const TraceLocalID &tlid,
                          const Mapper::MapTaskOutput &output,
                          const std::deque<InstanceSet> &physical_instances,
                          std::set<RtEvent> &applied_events);
      virtual void record_set_effects(Memoizable *memo, ApEvent &rhs);
      virtual void record_complete_replay(Memoizable *memo, ApEvent rhs);
      virtual void record_reservations(const TraceLocalID &tlid,
                                const std::map<Reservation,bool> &locks,
                                std::set<RtEvent> &applied_events);
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
      TraceInfo(SingleTask *task, RemoteTraceRecorder *rec, 
                bool initialize = false); 
      TraceInfo(const TraceInfo &info);
      ~TraceInfo(void);
    protected:
      TraceInfo(Operation *op, Memoizable *memo, 
                PhysicalTraceRecorder *rec, bool recording);
    public:
      inline void record_get_term_event(void) const
        {
          base_sanity_check();
          rec->record_get_term_event(memo);
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
      inline void record_merge_events(ApEvent &result, 
                                      const std::vector<ApEvent> &events) const
        {
          base_sanity_check();
          rec->record_merge_events(result, events, memo);
        }
      inline void record_op_sync_event(ApEvent &result) const
        {
          base_sanity_check();
          rec->record_set_op_sync_event(result, memo);
        }
      inline void record_mapper_output(const TraceLocalID &tlid, 
                          const Mapper::MapTaskOutput &output,
                          const std::deque<InstanceSet> &physical_instances,
                          std::set<RtEvent> &applied) const
        {
          base_sanity_check();
          rec->record_mapper_output(tlid, output, physical_instances, applied);
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
      inline void record_reservations(const TraceLocalID &tlid,
                      const std::map<Reservation,bool> &reservations,
                      std::set<RtEvent> &applied) const
        {
          base_sanity_check();
          rec->record_reservations(tlid, reservations, applied);
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
                          const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                          ApEvent precondition, PredEvent pred_guard) const
        {
          sanity_check();
          rec->record_issue_copy(memo, result, expr, src_fields, 
                                 dst_fields, reservations,
#ifdef LEGION_SPY
                                 src_tree_id, dst_tree_id,
#endif
                                 precondition, pred_guard);
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
      inline void record_post_fill_view(FillView *view, 
                                        const FieldMask &mask) const
        {
          sanity_check();
          rec->record_post_fill_view(view, mask);
        }
      inline void record_fill_views(ApEvent lhs,
                                    IndexSpaceExpression *expr, 
                                    const FieldMaskSet<FillView> &srcs,
                                    const FieldMaskSet<InstanceView> &dsts,
                                    std::set<RtEvent> &applied,
                                    bool reduction_initialization) const
        {
          sanity_check();
          rec->record_fill_views(lhs, memo, index, expr, 
                                 srcs, dsts, applied, reduction_initialization);
        }
      inline void record_issue_across(ApEvent &result, 
                                      ApEvent collective_precondition,
                                      ApEvent copy_precondition,
                                      ApEvent src_indirect_precondition,
                                      ApEvent dst_indirect_precondition,
                                      CopyAcrossExecutor *executor) const
        {
          sanity_check();
          rec->record_issue_across(memo, result, collective_precondition, 
                      copy_precondition, src_indirect_precondition,
                      dst_indirect_precondition, executor);
        }
      inline void record_copy_views(ApEvent lhs, unsigned idx1, unsigned idx2,
                                    PrivilegeMode mode1, PrivilegeMode mode2,
                                    IndexSpaceExpression *expr,
                                 const FieldMaskSet<InstanceView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                    bool src_indirect, bool dst_indirect,
                                    std::set<RtEvent> &applied) const
        {
          sanity_check();
          rec->record_copy_views(lhs, memo, idx1, idx2, expr,
                                 tracing_srcs, tracing_dsts, mode1, mode2,
                                 src_indirect, dst_indirect, applied);
        }
      inline void record_copy_views(ApEvent lhs,
                                    IndexSpaceExpression *expr,
                                 const FieldMaskSet<InstanceView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                    std::set<RtEvent> &applied) const
        {
          sanity_check();
          rec->record_copy_views(lhs, memo, index, dst_index, expr,
                                 tracing_srcs, tracing_dsts,
                                 LEGION_READ_PRIV, LEGION_WRITE_PRIV,
                                 false/*indirect*/, false/*indrect*/, applied);
        }
      inline void record_indirect_views(ApEvent indirect_done, ApEvent all_done,
                                        unsigned indirect_index,
                                        IndexSpaceExpression *expr,
                                        const FieldMaskSet<InstanceView> &views,
                                        std::set<RtEvent> &applied,
                                        PrivilegeMode privilege) const
        {
          sanity_check();
          rec->record_indirect_views(indirect_done, all_done, memo,
                   indirect_index, expr, views, applied, privilege);
        }
      inline void record_op_view(const RegionUsage &usage,
                                 const FieldMask &user_mask,
                                 InstanceView *view) const
        {
          sanity_check();
          rec->record_op_view(memo,index,view,usage,user_mask,update_validity);
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
                     IndexSpaceNode *launch_space);
    public:
      inline bool is_projecting(void) const { return (projection != NULL); }
    public:
      ProjectionFunction *projection;
      ProjectionType projection_type;
      IndexSpaceNode *projection_space;
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
                 bool dis, bool dirty_reduction = false);
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
      void merge(FieldState &rhs, RegionTreeNode *node);
      bool filter(const FieldMask &mask);
      void add_child(RegionTreeNode *child,
          const FieldMask &mask, std::set<RtEvent> &applied);
      void remove_child(RegionTreeNode *child);
    public:
      bool projection_domain_dominates(IndexSpaceNode *next_space) const;
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
      ProjectionFunction *projection;
      IndexSpaceNode *projection_space;
    };  

    /**
     * \class ProjectionEpoch
     * This class captures the set of projection functions
     * and domains that have performed in current open
     * projection epoch
     */
    class ProjectionEpoch : public LegionHeapify<ProjectionEpoch> {
    public:
      static const ProjectionEpochID first_epoch = 1;
    public:
      ProjectionEpoch(ProjectionEpochID epoch_id,
                      const FieldMask &mask);
      ProjectionEpoch(const ProjectionEpoch &rhs);
      ~ProjectionEpoch(void);
    public:
      ProjectionEpoch& operator=(const ProjectionEpoch &rhs);
    public:
      void insert(ProjectionFunction *function, IndexSpaceNode *space);
    public:
      const ProjectionEpochID epoch_id;
      FieldMask valid_fields;
    public:
      // For now we only record the write projections since we use them
      // for constructing composite view write sets
      std::map<ProjectionFunction*,
               std::set<IndexSpaceNode*> > write_projections;
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
    public:
      void advance_projection_epochs(const FieldMask &advance_mask);
      void update_projection_epochs(FieldMask capture_mask,
                                    const ProjectionInfo &info);
    public:
      RegionTreeNode *const owner;
    public:
      LegionList<FieldState,
                 LOGICAL_FIELD_STATE_ALLOC> field_states;
      LegionList<LogicalUser,CURR_LOGICAL_ALLOC> curr_epoch_users;
      LegionList<LogicalUser,PREV_LOGICAL_ALLOC> prev_epoch_users;
    public:
      // Keep track of which fields we've done a reduction to here
      FieldMask reduction_fields;
      LegionMap<ReductionOpID,FieldMask> outstanding_reductions;
    public:
      // Keep track of the current projection epoch for each field
      std::list<ProjectionEpoch*> projection_epochs;
    };

    typedef DynamicTableAllocator<LogicalState,10,8> LogicalStateAllocator;

    /**
     * \struct LogicalCloser
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
                                       const LogicalTraceInfo &trace_info);
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
      // At most we will ever generate three close operations at a node
      MergeCloseOp *close_op;
    protected:
      // Cache the generation IDs so we can kick off ops before adding users
      GenerationID merge_close_gen;
    }; 

    /**
     * \struct KDLine
     * A small helper struct for tracking splitting planes for 
     * KD-tree construction
     */
    struct KDLine {
    public:
      KDLine(void)
        : value(0), index(0), start(false) { }
      KDLine(coord_t val, unsigned idx, bool st)
        : value(val), index(idx), start(st) { }
    public:
      inline bool operator<(const KDLine &rhs) const
      {
        if (value < rhs.value)
          return true;
        if (value > rhs.value)
          return false;
        if (index < rhs.index)
          return true;
        if (index > rhs.index)
          return false;
        return start < rhs.start;
      }
      inline bool operator==(const KDLine &rhs) const
      {
        if (value != rhs.value)
          return false;
        if (index != rhs.index)
          return false;
        if (start != rhs.start)
          return false;
        return true;
      }
    public:
      coord_t value;
      unsigned index;
      bool start;
    };

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
      bool record_guard_set(EquivalenceSet *set);
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
               FieldMaskSet<IndexSpaceExpression> > InstanceFieldExprs;
      typedef LegionMap<ApEvent,
               FieldMaskSet<IndexSpaceExpression> > EventFieldExprs;
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
        virtual void compute_source_preconditions(RegionTreeForest *forest,
#ifdef DEBUG_LEGION
               const bool copy_across,
#endif
               const std::map<InstanceView*,EventFieldExprs> &src_pre,
               LegionMap<ApEvent,FieldMask> &preconditions) const = 0;
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
        virtual void compute_source_preconditions(RegionTreeForest *forest,
#ifdef DEBUG_LEGION
                   const bool copy_across,
#endif
                   const std::map<InstanceView*,EventFieldExprs> &src_pre,
                   LegionMap<ApEvent,FieldMask> &preconditions) const;
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
        virtual void compute_source_preconditions(RegionTreeForest *forest,
#ifdef DEBUG_LEGION
                   const bool copy_across,
#endif
                   const std::map<InstanceView*,EventFieldExprs> &src_pre,
                   LegionMap<ApEvent,FieldMask> &preconditions) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        FillView *const source;
      };
      typedef LegionMap<ApEvent,
               FieldMaskSet<Update> > EventFieldUpdates;
    public:
      CopyFillAggregator(RegionTreeForest *forest, Operation *op, 
                         unsigned idx, RtEvent guard_event, bool track_events,
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
      void record_updates(InstanceView *dst_view, 
                          const FieldMaskSet<LogicalView> &src_views,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      // Neither fills nor reductions should have a redop across as they
      // should have been applied an instance directly for across copies
      void record_fill(InstanceView *dst_view,
                       FillView *src_view,
                       const FieldMask &fill_mask,
                       IndexSpaceExpression *expr,
                       CopyAcrossHelper *across_helper = NULL);
      void record_reductions(InstanceView *dst_view,
                             const std::vector<ReductionView*> &src_views,
                             const unsigned src_fidx,
                             const unsigned dst_fidx,
                             IndexSpaceExpression *expr,
                             CopyAcrossHelper *across_helper = NULL);
      void record_reduction_fill(ReductionView *init_view,
                                 const FieldMask &fill_mask,
                                 IndexSpaceExpression *expr);
      // Record preconditions coming back from analysis on views
      void record_preconditions(InstanceView *view, bool reading,
                                EventFieldExprs &preconditions);
      void record_precondition(InstanceView *view, bool reading,
                               ApEvent event, const FieldMask &mask,
                               IndexSpaceExpression *expr);
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
      RtEvent perform_updates(const LegionMap<InstanceView*,
                            FieldMaskSet<Update> > &updates,
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
      LegionMap<InstanceView*,FieldMaskSet<Update> > sources; 
      std::vector</*vector over reduction epochs*/
        LegionMap<InstanceView*,FieldMaskSet<Update> > > reductions;
      // Figure out the reduction operator is for each epoch of a
      // given destination instance and field
      std::map<std::pair<InstanceView*,unsigned/*dst fidx*/>,
               std::vector<ReductionOpID> > reduction_epochs;
      std::set<LogicalView*> all_views; // used for reference counting
    protected:
      mutable LocalLock pre_lock; 
      std::map<InstanceView*,EventFieldExprs> dst_pre, src_pre;
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
      std::map<InstanceView*,LegionVector<SourceQuery> > mapper_queries;
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
         const FieldMask &mask, RtUserEvent done, bool cached_set,
         bool already_deferred = true);
      public:
        PhysicalAnalysis *const analysis;
        EquivalenceSet *const set;
        FieldMask *const mask;
        const RtUserEvent applied_event;
        const RtUserEvent done_event;
        const bool cached_set;
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
                       const VersionInfo &info, bool on_heap);
      // Remote physical analysis
      PhysicalAnalysis(Runtime *rt, AddressSpaceID source, AddressSpaceID prev,
             Operation *op, unsigned index, VersionManager *man, bool on_heap);
      PhysicalAnalysis(const PhysicalAnalysis &rhs);
      virtual ~PhysicalAnalysis(void);
    public:
      inline bool has_remote_sets(void) const
        { return !remote_sets.empty(); }
      inline void record_parallel_traversals(void)
        { parallel_traversals = true; } 
    public:
      void traverse(EquivalenceSet *set, const FieldMask &mask, 
                    std::set<RtEvent> &deferral_events,
                    std::set<RtEvent> &applied_events, const bool cached_set,
                    RtEvent precondition = RtEvent::NO_RT_EVENT,
                    const bool already_deferred = false);
      void defer_traversal(RtEvent precondition, EquivalenceSet *set,
              const FieldMask &mask, std::set<RtEvent> &deferral_events,
              std::set<RtEvent> &applied_events, const bool cached_set,
              RtUserEvent deferral_event = RtUserEvent::NO_RT_USER_EVENT,
              const bool already_deferred = true);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
      void process_local_instances(const FieldMaskSet<InstanceView> &views,
                                   const bool local_restricted);
      void filter_remote_expressions(FieldMaskSet<IndexSpaceExpression> &exprs);
      // Return true if any are restricted
      bool report_instances(FieldMaskSet<InstanceView> &instances);
    public:
      // Lock taken by these methods if needed
      bool update_alt_sets(EquivalenceSet *set, FieldMask &mask);
      void filter_alt_sets(EquivalenceSet *set, const FieldMask &mask);
      void record_stale_set(EquivalenceSet *set, const FieldMask &mask);
      void record_remote(EquivalenceSet *set, const FieldMask &mask,
                         const AddressSpaceID owner, bool cached_set);
    public:
      // Lock must be held from caller
      void record_instance(InstanceView* view, const FieldMask &mask);
      inline void record_restriction(void) { restricted = true; }
    protected:
      // Can only be called once all traversals are done
      void update_stale_equivalence_sets(std::set<RtEvent> &applied_events);
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
      Operation *const op;
      const unsigned index;
      VersionManager *const version_manager;
      const bool owns_op;
      const bool on_heap;
    protected:
      // For updates to the traversal data structures
      FieldMaskSet<EquivalenceSet> alt_sets;
      FieldMaskSet<EquivalenceSet> stale_sets;
    protected:
      LegionMap<std::pair<AddressSpaceID,bool>,
                FieldMaskSet<EquivalenceSet> > remote_sets;
      FieldMaskSet<InstanceView> *remote_instances;
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
                        const VersionInfo &info, ReductionOpID redop = 0);
      ValidInstAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                        Operation *op, unsigned index, VersionManager *man,
                        ValidInstAnalysis *target, ReductionOpID redop);
      ValidInstAnalysis(const ValidInstAnalysis &rhs);
      virtual ~ValidInstAnalysis(void);
    public:
      ValidInstAnalysis& operator=(const ValidInstAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
                                public LegionHeapify<ValidInstAnalysis> {
    public:
      InvalidInstAnalysis(Runtime *rt, Operation *op, unsigned index,
                          const VersionInfo &info,
                          const FieldMaskSet<InstanceView> &valid_instances);
      InvalidInstAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                          Operation *op, unsigned index, VersionManager *man, 
                          InvalidInstAnalysis *target, 
                          const FieldMaskSet<InstanceView> &valid_instances);
      InvalidInstAnalysis(const InvalidInstAnalysis &rhs);
      virtual ~InvalidInstAnalysis(void);
    public:
      InvalidInstAnalysis& operator=(const InvalidInstAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
      const FieldMaskSet<InstanceView> valid_instances;
      InvalidInstAnalysis *const target_analysis;
    };

    /**
     * \class UpdateAnalysis
     * For performing updates on equivalence set trees
     */
    class UpdateAnalysis : public PhysicalAnalysis,
                           public LegionHeapify<UpdateAnalysis> {
    public:
      UpdateAnalysis(Runtime *rt, Operation *op, unsigned index,
                     const VersionInfo &info, const RegionRequirement &req,
                     RegionNode *node, const InstanceSet &target_instances,
                     std::vector<InstanceView*> &target_views,
                     const PhysicalTraceInfo &trace_info,
                     const ApEvent precondition, const ApEvent term_event,
                     const bool check_initialized, const bool record_valid);
      UpdateAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                     Operation *op, unsigned index, VersionManager *man,
                     const RegionUsage &usage, RegionNode *node, 
                     InstanceSet &target_instances,
                     std::vector<InstanceView*> &target_views,
                     const PhysicalTraceInfo &trace_info,
                     const RtEvent user_registered,
                     const ApEvent precondition, const ApEvent term_event,
                     const bool check_initialized, const bool record_valid);
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
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
      const RegionUsage usage;
      RegionNode *const node;
      const InstanceSet target_instances;
      const std::vector<InstanceView*> target_views;
      const PhysicalTraceInfo trace_info;
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
                      const VersionInfo &info);
      AcquireAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                      Operation *op, unsigned index, VersionManager *man,
                      AcquireAnalysis *target); 
      AcquireAnalysis(const AcquireAnalysis &rhs);
      virtual ~AcquireAnalysis(void);
    public:
      AcquireAnalysis& operator=(const AcquireAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
                      ApEvent precondition, const VersionInfo &info,
                      const PhysicalTraceInfo &trace_info);
      ReleaseAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                      Operation *op, unsigned index, VersionManager *manager,
                      ApEvent precondition, ReleaseAnalysis *target, 
                      const PhysicalTraceInfo &info);
      ReleaseAnalysis(const ReleaseAnalysis &rhs);
      virtual ~ReleaseAnalysis(void);
    public:
      ReleaseAnalysis& operator=(const ReleaseAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
      CopyAcrossAnalysis(Runtime *rt, Operation *op, unsigned src_index,
                         unsigned dst_index, const VersionInfo &info,
                         const RegionRequirement &src_req,
                         const RegionRequirement &dst_req,
                         const InstanceSet &target_instances,
                         const std::vector<InstanceView*> &target_views,
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
      // No perform_traversal here since we also need an
      // index space expression to perform the traversal
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
                        const RegionUsage &usage,
                        const VersionInfo &info, LogicalView *view,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const RtEvent guard_event = RtEvent::NO_RT_EVENT,
                        const PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
                        const bool track_effects = false,
                        const bool add_restriction = false);
      // Also local but with a full set of views
      OverwriteAnalysis(Runtime *rt, Operation *op, unsigned index,
                        const RegionUsage &usage,
                        const VersionInfo &info, 
                        const std::set<LogicalView*> &views,
                        const PhysicalTraceInfo &trace_info,
                        const ApEvent precondition,
                        const RtEvent guard_event = RtEvent::NO_RT_EVENT,
                        const PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
                        const bool track_effects = false,
                        const bool add_restriction = false);
      OverwriteAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                        Operation *op, unsigned index, VersionManager *man, 
                        const RegionUsage &usage, 
                        const std::set<LogicalView*> &views,
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
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
      const std::set<LogicalView*> views;
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
                     const VersionInfo &info, InstanceView *inst_view,
                     LogicalView *registration_view,
                     const bool remove_restriction = false);
      FilterAnalysis(Runtime *rt, AddressSpaceID src, AddressSpaceID prev,
                     Operation *op, unsigned index, VersionManager *man,
                     InstanceView *inst_view, LogicalView *registration_view,
                     const bool remove_restriction);
      FilterAnalysis(const FilterAnalysis &rhs);
      virtual ~FilterAnalysis(void);
    public:
      FilterAnalysis& operator=(const FilterAnalysis &rhs);
    public:
      virtual void perform_traversal(EquivalenceSet *set,
                                     const FieldMask &mask,
                                     std::set<RtEvent> &deferral_events,
                                     std::set<RtEvent> &applied_events,
                                     FieldMask *stale_mask,
                                     const bool cached_set,
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
     * \class RayTracer
     * This is an abstract class that provides an interface for
     * recording the equivalence sets that result from ray tracing
     * an equivalence set tree for a given index space expression.
     */
    class RayTracer {
    public:
      virtual ~RayTracer(void) { }
    public:
      virtual void record_equivalence_set(EquivalenceSet *set,
                                          const FieldMask &mask) = 0;
      virtual void record_pending_equivalence_set(EquivalenceSet *set,
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
                          // These are just for the case where the
                          // request comes from a remote node and
                          // we're waiting for the expression to load
                          const PendingRemoteExpression *pending = NULL);
      public:
        EquivalenceSet *const set;
        RayTracer *const target;
        IndexSpaceExpression *const expr;
        const IndexSpace handle;
        const AddressSpaceID origin;
        const RtUserEvent done;
        const RtUserEvent deferral;
        FieldMask *const ray_mask;
        const PendingRemoteExpression *const pending;
      };
      struct DeferRayTraceFinishArgs : 
        public LgTaskArgs<DeferRayTraceFinishArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_RAY_TRACE_FINISH_TASK_ID;
      public:
        DeferRayTraceFinishArgs(RayTracer *t, AddressSpaceID src,
            FieldMaskSet<EquivalenceSet> *to_tv,
            std::map<EquivalenceSet*,IndexSpaceExpression*> *exs,
            const size_t v, const IndexSpace h, RtUserEvent d)
          : LgTaskArgs<DeferRayTraceFinishArgs>(implicit_provenance),
            target(t), source(src), to_traverse(to_tv), exprs(exs), 
            volume(v), handle(h), done(d) { }
      public:
        RayTracer *const target;
        const AddressSpaceID source;
        FieldMaskSet<EquivalenceSet> *const to_traverse;
        std::map<EquivalenceSet*,IndexSpaceExpression*> *const exprs;
        const size_t volume;
        const IndexSpace handle;
        const RtUserEvent done;
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
            LegionMap<VersionID,FieldMask> *u, RtUserEvent d)
          : LgTaskArgs<DeferMergeOrForwardArgs>(implicit_provenance),
            set(s), views(v), reductions(r), restricted(t), 
            versions(u), done(d), initial(init) { }
      public:
        EquivalenceSet *const set;
        FieldMaskSet<LogicalView> *const views;
        std::map<unsigned,std::vector<ReductionView*> > *const reductions;
        FieldMaskSet<InstanceView> *const restricted;
        LegionMap<VersionID,FieldMask> *const versions;
        const RtUserEvent done;
        const bool initial;
      };
      struct DeferResponseArgs : public LgTaskArgs<DeferResponseArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_EQ_RESPONSE_TASK_ID;
      public:
        DeferResponseArgs(DistributedID id, AddressSpaceID src, 
                          AddressSpaceID log, IndexSpaceExpression *ex, 
                          const PendingRemoteExpression &pending, IndexSpace h);
      public:
        const DistributedID did;
        const AddressSpaceID source;
        const AddressSpaceID logical_owner;
        IndexSpaceExpression *const expr;
        const PendingRemoteExpression *const pending;
        const IndexSpace handle;
      };
      struct DeferRemoveRefArgs : public LgTaskArgs<DeferRemoveRefArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_REMOVE_EQ_REF_TASK_ID;
      public:
        DeferRemoveRefArgs(std::vector<IndexSpaceExpression*> *refs,
                           DistributedID src)
          : LgTaskArgs<DeferRemoveRefArgs>(implicit_provenance),
            references(refs), source(src) { }
      public:
        std::vector<IndexSpaceExpression*> *const references;
        const DistributedID source;
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
                                    std::set<RtEvent> &applied_events);
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
          const LegionMap<VersionID,FieldMask> &new_versions);
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
        RtEvent continuation_pre;
        continuation_pre.id =
          next_deferral_precondition.exchange(deferral_event.id);
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
      LegionMap<VersionID,FieldMask>                      version_numbers;
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
      std::atomic<Realm::Event::id_t>                next_deferral_precondition;
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

    /**
     * \class VersionManager
     * The VersionManager class tracks the starting equivalence
     * sets for a given node in the logical region tree. Note
     * that its possible that these have since been shattered
     * and we need to traverse them, but it's a cached starting
     * point that doesn't involve tracing the entire tree.
     */
    class VersionManager : public RayTracer, 
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
      void reset(void);
    public:
      RtEvent perform_versioning_analysis(InnerContext *parent_ctx,
                                          VersionInfo *version_info,
                                          RegionNode *region_node,
                                          const FieldMask &version_mask,
                                          Operation *op);
      virtual void record_equivalence_set(EquivalenceSet *set, 
                                          const FieldMask &mask);
      virtual void record_pending_equivalence_set(EquivalenceSet *set, 
                                          const FieldMask &mask);
      void finalize_equivalence_sets(RtUserEvent done_event);                           
      RtEvent record_stale_sets(FieldMaskSet<EquivalenceSet> &stale_sets);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                                TreeStateLogger *logger);
    public:
      static void handle_finalize_eq_sets(const void *args);
      static void handle_stale_update(Deserializer &derez, Runtime *runtime);
    public:
      const ContextID ctx;
      RegionTreeNode *const node;
      Runtime *const runtime;
    protected:
      mutable LocalLock manager_lock;
    protected: 
      FieldMaskSet<EquivalenceSet> equivalence_sets;
      FieldMaskSet<EquivalenceSet> pending_equivalence_sets;
      FieldMaskSet<VersionInfo> waiting_infos;
      LegionMap<RtUserEvent,FieldMask> equivalence_sets_ready;
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
      LegionMap<unsigned/*depth*/,FieldMask> interfering_children;
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
     * \class LogicalPathRegistrar
     * A class that registers dependences for an operation
     * against all other operation with an overlapping
     * field mask along a given path
     */
    class LogicalPathRegistrar : public PathTraverser {
    public:
      LogicalPathRegistrar(ContextID ctx, Operation *op,
            const FieldMask &field_mask, RegionTreePath &path);
      LogicalPathRegistrar(const LogicalPathRegistrar &rhs);
      virtual ~LogicalPathRegistrar(void);
    public:
      LogicalPathRegistrar& operator=(const LogicalPathRegistrar &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const FieldMask field_mask;
      Operation *const op;
    };

    /**
     * \class LogicalRegistrar
     * A class that registers dependences for an operation
     * against all other operations with an overlapping
     * field mask.
     */
    class LogicalRegistrar : public NodeTraverser {
    public:
      LogicalRegistrar(ContextID ctx, Operation *op,
                       const FieldMask &field_mask,
                       bool dom);
      LogicalRegistrar(const LogicalRegistrar &rhs);
      ~LogicalRegistrar(void);
    public:
      LogicalRegistrar& operator=(const LogicalRegistrar &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const FieldMask field_mask;
      Operation *const op;
      const bool dominate;
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

    /**
     * \class VersioningInvalidator
     * A class for reseting the versioning managers for 
     * a deleted region (sub)-tree so that version states
     * and the things they point to can be cleaned up
     * by the garbage collector. The better long term
     * answer is to have individual contexts do this.
     */
    class VersioningInvalidator : public NodeTraverser {
    public:
      VersioningInvalidator(void);
      VersioningInvalidator(RegionTreeContext ctx);
    public:
      virtual bool visit_only_valid(void) const { return true; }
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const bool invalidate_all;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_ANALYSIS_H__
