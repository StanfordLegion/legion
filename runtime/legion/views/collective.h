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

#ifndef __LEGION_COLLECTIVE_VIEW_H__
#define __LEGION_COLLECTIVE_VIEW_H__

#include "legion/views/instance.h"
#include "legion/instances/instance.h"
#include "legion/tools/types.h"
#include "legion/tracing/recording.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CollectiveView
     * This class provides an abstract base class for any kind of view
     * that represents a group of instances that need to be analyzed
     * cooperatively for physical analysis.
     */
    class CollectiveView : public InstanceView,
                           public InstanceDeletionSubscriber {
    public:
      enum ValidState {
        FULL_VALID_STATE,
        PENDING_INVALID_STATE,
        NOT_VALID_STATE,
      };
    public:
      CollectiveView(
          DistributedID did, DistributedID context_did,
          const std::vector<IndividualView*>& views,
          const std::vector<DistributedID>& instances, bool register_now,
          CollectiveMapping* mapping);
      virtual ~CollectiveView(void);
    public:
      virtual AddressSpaceID get_analysis_space(
          PhysicalManager* inst) const override;
      virtual bool aliases(InstanceView* other) const override;
    public:
      // Reference counting state change functions
      virtual void notify_local(void) override;
      virtual void notify_valid(void) override;
      virtual bool notify_invalid(void) override;
    public:
      virtual void pack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
          override;
      virtual void unpack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
          override;
    public:
      virtual ApEvent fill_from(
          FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& fill_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool fill_restricted, const bool need_valid_return) override;
      virtual ApEvent copy_from(
          InstanceView* src_view, ApEvent precondition,
          PredEvent predicate_guard, ReductionOpID redop,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& copy_mask,
          PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool copy_restricted, const bool need_valid_return) override;
      virtual ApEvent register_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, ApEvent term_event, PhysicalManager* target,
          CollectiveMapping* collective_mapping,
          size_t local_collective_arrivals,
          std::vector<RtEvent>& registered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const AddressSpaceID source,
          const bool symbolic = false) override;
      // This is a special entry point variation copy_from only for
      // collective view (not it is not virtual) that will handle the
      // special case where we have a bunch of individual views that
      // we'll be copying to this collective view, so we can do all
      // the individual copies to a local instance, and then fuse the
      // resulting broadcast or reduce out to everywhere
      ApEvent collective_fuse_gather(
          const std::map<IndividualView*, IndexSpaceExpression*>& sources,
          ApEvent precondition, PredEvent predicate_guard, Operation* op,
          const unsigned index, const IndexSpace collective_match_space,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          const bool copy_restricted, const bool need_valid_return);
    public:
      void perform_collective_fill(
          FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace match_space, const size_t op_context_index,
          const FieldMask& fill_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent result, AddressSpaceID origin,
          const bool fill_restricted);
      ApEvent perform_collective_point(
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expresison,
          IndexSpace upper_bound, Operation* op, const unsigned index,
          const FieldMask& copy_mask, const FieldMask& dst_mask,
          const Memory location, const UniqueInst& dst_inst,
          const LgEvent dst_unique_event, const DistributedID src_inst_did,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CollectiveKind collective = COLLECTIVE_NONE);
      void perform_collective_broadcast(
          const std::vector<CopySrcDstField>& src_fields, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expresison,
          Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const size_t op_ctx_index,
          const FieldMask& copy_mask, const UniqueInst& src_inst,
          const LgEvent src_unique_event, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent copy_done, ApUserEvent all_done, ApBarrier all_bar,
          ShardID owner_shard, AddressSpaceID origin,
          const bool copy_restricted, const CollectiveKind collective_kind);
      void perform_collective_reducecast(
          ReductionView* source, const std::vector<CopySrcDstField>& src_fields,
          ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* copy_expresison, Operation* op,
          const unsigned index, const IndexSpace collective_match_space,
          const size_t op_ctx_index, const FieldMask& copy_mask,
          const UniqueInst& src_inst, const LgEvent src_unique_event,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent copy_done, ApBarrier all_bar, ShardID owner_shard,
          AddressSpaceID origin, const bool copy_restricted);
      void perform_collective_hourglass(
          AllreduceView* source, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expresison,
          Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& copy_mask,
          const DistributedID src_inst_did, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent all_done, AddressSpaceID target,
          const bool copy_restricted);
      void perform_collective_pointwise(
          CollectiveView* source, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
          Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const size_t op_ctx_index,
          const FieldMask& copy_mask, const DistributedID src_inst_did,
          const UniqueID src_inst_op_id, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent all_done, ApBarrier all_bar, ShardID owner_shard,
          AddressSpaceID origin, const uint64_t allreduce_tag,
          const bool copy_restricted);
    public:
      inline AddressSpaceID select_origin_space(void) const
      {
        return (
            collective_mapping->contains(local_space) ?
                local_space :
                collective_mapping->find_nearest(local_space));
      }
      bool contains(PhysicalManager* manager) const;
      bool meets_regions(
          const std::vector<LogicalRegion>& regions,
          bool tight_bounds = false) const;
      void find_instances_in_memory(
          Memory memory, std::vector<PhysicalManager*>& instances);
      void find_instances_nearest_memory(
          Memory memory, std::vector<PhysicalManager*>& instances,
          bool bandwidth);
    public:
      void process_remote_instances_response(
          AddressSpaceID source, const std::vector<IndividualView*>& view);
      void record_remote_instances(const std::vector<IndividualView*>& view);
      RtEvent find_instances_nearest_memory(
          Memory memory, AddressSpaceID source,
          std::vector<DistributedID>* instances, std::atomic<size_t>* target,
          AddressSpaceID origin, size_t best, bool bandwidth);
      void find_nearest_local_instances(
          Memory memory, size_t& best, std::vector<PhysicalManager*>& results,
          bool bandwidth) const;
    public:
      AddressSpaceID select_source_space(AddressSpaceID destination) const;
      void pack_fields(
          Serializer& rez, const std::vector<CopySrcDstField>& fields,
          LgEvent inst_uid) const;
      unsigned find_local_index(PhysicalManager* target) const;
      void register_collective_analysis(
          PhysicalManager* target, CollectiveAnalysis* analysis,
          std::set<RtEvent>& applied_events);
    public:
      void notify_instance_deletion(RegionTreeID tid);
      virtual void notify_instance_deletion(PhysicalManager* manager) override;
      virtual void add_subscriber_reference(PhysicalManager* manager) override;
      virtual bool remove_subscriber_reference(
          PhysicalManager* manager) override;
    public:
      void process_register_user_request(
          const size_t op_ctx_index, const unsigned index,
          const IndexSpace match_space, RtEvent registered, RtEvent applied);
      void process_register_user_response(
          const size_t op_ctx_index, const unsigned index,
          const IndexSpace match_space, const RtEvent registered,
          const RtEvent applied);
    protected:
      ApEvent register_collective_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, ApEvent term_event, PhysicalManager* target,
          size_t local_collective_arrivals,
          std::vector<RtEvent>& regsitered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const bool symbolic);
      void finalize_collective_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, RtUserEvent local_registered,
          RtEvent global_registered, RtUserEvent local_applied,
          RtEvent global_applied, std::vector<ApUserEvent>& ready_events,
          std::vector<std::vector<ApEvent> >& terms,
          const PhysicalTraceInfo* trace_info, const bool symbolic);
      void perform_local_broadcast(
          IndividualView* local_view,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<AddressSpaceID>& children,
          CollectiveAnalysis* first_local_analysis, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expresison,
          Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const size_t op_ctx_index,
          const FieldMask& copy_mask, const UniqueInst& src_inst,
          const LgEvent src_unique_event, const PhysicalTraceInfo& local_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent all_done, ApBarrier all_bar, ShardID owner_shard,
          AddressSpaceID origin, const bool copy_restricted,
          const CollectiveKind collective_kind);
    protected:
      void broadcast_local(
          const PhysicalManager* src_manager, const unsigned src_index,
          Operation* op, const unsigned index, IndexSpaceExpression* copy_expr,
          IndexSpace match_space, const FieldMask& copy_mask,
          ApEvent precondition, PredEvent predicate_guard,
          const std::vector<CopySrcDstField>& src_fields,
          const UniqueInst& src_inst, const PhysicalTraceInfo& trace_info,
          const CollectiveKind collective_kind,
          std::vector<ApEvent>& destination_events,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          const bool has_instance_events = false,
          const bool first_local_analysis = false,
          const size_t op_ctx_index = 0);
      const std::vector<std::pair<unsigned, unsigned> >&
          find_spanning_broadcast_copies(unsigned root_index);
      bool construct_spanning_adjacency_matrix(
          unsigned root_index,
          const std::map<Memory, unsigned>& first_in_memory,
          std::vector<float>& adjacency_matrix) const;
      void compute_spanning_tree_same_bandwidth(
          unsigned root_index, const std::vector<float>& adjacency_matrix,
          std::vector<unsigned>& previous,
          std::map<Memory, unsigned>& first_in_memory) const;
      void compute_spanning_tree_diff_bandwidth(
          unsigned root_index, const std::vector<float>& adjacency_matrix,
          std::vector<unsigned>& previous,
          std::map<Memory, unsigned>& first_in_memory) const;
    public:
      void make_valid(bool need_lock);
      bool make_invalid(bool need_lock);
      bool perform_invalidate_request(LamportClock snapshot, bool need_lock);
      bool perform_invalidate_response(
          LamportClock snapshot, uint64_t sent, uint64_t received,
          LamportClock clock, bool failed, bool need_lock);
    public:
      static void process_nearest_instances(
          std::atomic<size_t>* target, std::vector<DistributedID>* instances,
          size_t best, const std::vector<DistributedID>& results,
          bool bandwidth);
      [[nodiscard]] static LgEvent unpack_fields(
          std::vector<CopySrcDstField>& fields, Deserializer& derez,
          std::set<RtEvent>& ready_events, CollectiveView* view,
          RtEvent view_ready);
      static bool has_multiple_local_memories(
          const std::vector<IndividualView*>& local_views);
    public:
      const DistributedID context_did;
      const std::vector<DistributedID> instances;
      const std::vector<IndividualView*> local_views;
    protected:
      std::map<PhysicalManager*, IndividualView*> remote_instances;
      NodeSet<LONG_LIFETIME> remote_instance_responses;
    protected:
      struct UserRendezvous {
        UserRendezvous(void)
          : remaining_local_arrivals(0), remaining_remote_arrivals(0),
            remaining_analyses(0), trace_info(nullptr), mask(nullptr),
            expr(nullptr), op_id(0), symbolic(false), local_initialized(false)
        { }
        // event for when local instances can be used
        std::vector<ApUserEvent> ready_events;
        // all the local term events for each view
        std::vector<std::vector<ApEvent> > local_term_events;
        // events from remote nodes indicating they are registered
        std::vector<RtEvent> remote_registered;
        // events from remote nodes indicating they are applied
        std::vector<RtEvent> remote_applied;
        // event to trigger when local registration is done
        RtUserEvent local_registered;
        // event that marks when all registrations are done
        RtUserEvent global_registered;
        // event to trigger when local effects are done
        RtUserEvent local_applied;
        // event that marks when all effects are done
        RtUserEvent global_applied;
        // Counts of remaining notficiations before registration
        unsigned remaining_local_arrivals;
        unsigned remaining_remote_arrivals;
        unsigned remaining_analyses;
        // PhysicalTraceInfo that made the ready_event and should trigger it
        PhysicalTraceInfo* trace_info;
        // Arguments for performing the local registration
        RegionUsage usage;
        HeapifyBox<FieldMask, OPERATION_LIFETIME>* mask;
        IndexSpaceNode* expr;
        UniqueID op_id;
        bool symbolic;
        bool local_initialized;
      };
      std::map<RendezvousKey, UserRendezvous> rendezvous_users;
    private:
      // For valid state tracking
      ValidState valid_state;
      uint32_t remaining_invalidation_responses;
      // Lamport clock for this view's valid-reference invalidation protocol.
      // collect_lamport_clock is the view's logical clock (stamped onto packed
      // valid references, merged on unpack and on invalidate responses).
      // pending_collect_lamport_clock is the in-progress round's snapshot and
      // also the round identifier (it replaces the old invalidation generation
      // counter). bump_collect_lamport_clock arms the clock so the first valid
      // reference packed after we commit to a round advances it past the
      // snapshot, which lets the owner detect a causality violation that would
      // otherwise hide behind aliased sent/received counts.
      LamportClock collect_lamport_clock;
      LamportClock pending_collect_lamport_clock;
      bool bump_collect_lamport_clock;
      uint64_t total_valid_sent, total_valid_received;
      uint64_t sent_valid_references, received_valid_references;
      bool invalidation_failed;
    private:
      // Use this flag to deduplicate deletion notifications from our instances
      std::atomic<bool> deletion_notified;
    protected:
      // Whether our local views are contained in multiple local memories
      const bool multiple_local_memories;
      std::map<unsigned, std::vector<std::pair<unsigned, unsigned> > >
          spanning_copies;
    };

    //--------------------------------------------------------------------------
    inline CollectiveView* LogicalView::as_collective_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_collective_view());
      return static_cast<CollectiveView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COLLECTIVE_VIEW_H__
