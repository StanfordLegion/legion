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

#ifndef __LEGION_COPY_FILL_AGGREGATOR_H__
#define __LEGION_COPY_FILL_AGGREGATOR_H__

#include "legion/kernel/metatask.h"
#include "legion/tracing/recording.h"
#include "legion/utilities/fieldmask_map.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CopyFillGuard
     * This is the base class for copy fill guards. It serves as a way to
     * ensure that multiple readers that can race to update an equivalence
     * set observe each others changes before performing their copies.
     */
    class CopyFillGuard {
    public:
      struct CopyFillDeletion : public LgTaskArgs<CopyFillDeletion> {
      public:
        static constexpr LgTaskID TASK_ID = LG_COPY_FILL_DELETION_TASK_ID;
      public:
        CopyFillDeletion(void) = default;
        CopyFillDeletion(CopyFillGuard* g, RtUserEvent r)
          : LgTaskArgs<CopyFillDeletion>(false, true), guard(g), released(r)
        { }
      public:
        void execute(void) const;
      public:
        CopyFillGuard* guard;
        RtUserEvent released;
      };
    public:
#ifndef NON_AGGRESSIVE_AGGREGATORS
      CopyFillGuard(RtUserEvent post, RtUserEvent applied);
#else
      CopyFillGuard(RtUserEvent applied);
#endif
      CopyFillGuard(const CopyFillGuard& rhs) = delete;
      virtual ~CopyFillGuard(void);
    public:
      CopyFillGuard& operator=(const CopyFillGuard& rhs) = delete;
    public:
      void pack_guard(Serializer& rez);
      static CopyFillGuard* unpack_guard(
          Deserializer& derez, EquivalenceSet* set);
    public:
      bool record_guard_set(EquivalenceSet* set, bool read_only_guard);
      bool release_guards(
          std::set<RtEvent>& applied, bool force_deferral = false);
      static void handle_deletion(CopyFillDeletion& args);
    private:
      void release_guarded_sets(std::set<RtEvent>& released);
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
    class CopyFillAggregator
      : public CopyFillGuard,
        public Heapify<CopyFillAggregator, OPERATION_LIFETIME> {
    public:
      struct CopyFillAggregation : public LgTaskArgs<CopyFillAggregation> {
      public:
        static constexpr LgTaskID TASK_ID = LG_COPY_FILL_AGGREGATION_TASK_ID;
      public:
        CopyFillAggregation(void) = default;
        CopyFillAggregation(
            CopyFillAggregator* a, const PhysicalTraceInfo& i, ApEvent p,
            const bool manage_dst, const bool restricted, int s,
            std::map<InstanceView*, std::vector<ApEvent> >* dsts)
          : LgTaskArgs<CopyFillAggregation>(false, false),
            dst_events(
                (dsts == nullptr) ?
                    nullptr :
                    new std::map<InstanceView*, std::vector<ApEvent> >()),
            aggregator(a), info(new PhysicalTraceInfo(i)), pre(p), stage(s),
            manage_dst_events(manage_dst), restricted_output(restricted)
        {
          if (dsts != nullptr)
            dst_events->swap(*dsts);
        }
      public:
        void execute(void) const;
      public:
        std::map<InstanceView*, std::vector<ApEvent> >* dst_events;
        CopyFillAggregator* aggregator;
        PhysicalTraceInfo* info;
        ApEvent pre;
        int stage;
        bool manage_dst_events;
        bool restricted_output;
      };
    public:
      typedef op::map<InstanceView*, op::FieldMaskMap<IndexSpaceExpression> >
          InstanceFieldExprs;
      typedef op::map<ApEvent, FieldMask> EventFieldMap;
      class CopyUpdate;
      class FillUpdate;
      class Update {
      public:
        Update(
            IndexSpaceExpression* exp, const FieldMask& mask,
            CopyAcrossHelper* helper);
        virtual ~Update(void);
      public:
        virtual void record_source_expressions(
            InstanceFieldExprs& src_exprs) const = 0;
        virtual void sort_updates(
            std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
            std::vector<FillUpdate*>& fills) = 0;
      public:
        IndexSpaceExpression* const expr;
        const FieldMask src_mask;
        CopyAcrossHelper* const across_helper;
      };
      class CopyUpdate : public Update,
                         public Heapify<CopyUpdate, OPERATION_LIFETIME> {
      public:
        CopyUpdate(
            InstanceView* src, PhysicalManager* man, const FieldMask& mask,
            IndexSpaceExpression* expr, ReductionOpID red = 0,
            CopyAcrossHelper* helper = nullptr)
          : Update(expr, mask, helper), source(src), src_man(man), redop(red)
        { }
        virtual ~CopyUpdate(void) { }
      private:
        CopyUpdate(const CopyUpdate& rhs) = delete;
        CopyUpdate& operator=(const CopyUpdate& rhs) = delete;
      public:
        virtual void record_source_expressions(
            InstanceFieldExprs& src_exprs) const;
        virtual void sort_updates(
            std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
            std::vector<FillUpdate*>& fills);
      public:
        InstanceView* const source;
        PhysicalManager* const src_man;  // which source manager for collectives
        const ReductionOpID redop;
      };
      class FillUpdate : public Update,
                         public Heapify<FillUpdate, OPERATION_LIFETIME> {
      public:
        FillUpdate(
            FillView* src, const FieldMask& mask, IndexSpaceExpression* expr,
            PredEvent guard, CopyAcrossHelper* helper = nullptr)
          : Update(expr, mask, helper), source(src), fill_guard(guard)
        { }
        virtual ~FillUpdate(void) { }
      private:
        FillUpdate(const FillUpdate& rhs) = delete;
        FillUpdate& operator=(const FillUpdate& rhs) = delete;
      public:
        virtual void record_source_expressions(
            InstanceFieldExprs& src_exprs) const;
        virtual void sort_updates(
            std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
            std::vector<FillUpdate*>& fills);
      public:
        FillView* const source;
        // Unlike copies, because of nested predicated fill operations,
        // then fills can have their own predication guard different
        // from the base predicate guard for the aggregator
        const PredEvent fill_guard;
      };
      typedef op::map<ApEvent, op::FieldMaskMap<Update> > EventFieldUpdates;
    public:
      CopyFillAggregator(
          PhysicalAnalysis* analysis, CopyFillGuard* previous,
          bool track_events, PredEvent pred_guard = PredEvent::NO_PRED_EVENT);
      CopyFillAggregator(
          PhysicalAnalysis* analysis, unsigned src_index, unsigned dst_idx,
          CopyFillGuard* previous, bool track_events,
          PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
          // Used only in the case of copy-across analyses
          RtEvent alternate_pre = RtEvent::NO_RT_EVENT);
      CopyFillAggregator(const CopyFillAggregator& rhs) = delete;
      virtual ~CopyFillAggregator(void);
    public:
      CopyFillAggregator& operator=(const CopyFillAggregator& rhs) = delete;
    public:
      void record_update(
          InstanceView* dst_view, PhysicalManager* dst_man,
          LogicalView* src_view, const FieldMask& src_mask,
          IndexSpaceExpression* expr, const PhysicalTraceInfo& trace_info,
          EquivalenceSet* tracing_eq, ReductionOpID redop = 0,
          CopyAcrossHelper* across_helper = nullptr);
      void record_updates(
          InstanceView* dst_view, PhysicalManager* dst_man,
          const FieldMapView<LogicalView>& src_views, const FieldMask& src_mask,
          IndexSpaceExpression* expr, const PhysicalTraceInfo& trace_info,
          EquivalenceSet* tracing_eq, ReductionOpID redop = 0,
          CopyAcrossHelper* across_helper = nullptr);
      void record_partial_updates(
          InstanceView* dst_view, PhysicalManager* dst_man,
          const MapView<
              LogicalView*, local::FieldMaskMap<IndexSpaceExpression> >&
              src_views,
          const FieldMask& src_mask, IndexSpaceExpression* expr,
          const PhysicalTraceInfo& trace_info, EquivalenceSet* tracing_eq,
          ReductionOpID redop = 0, CopyAcrossHelper* across_helper = nullptr);
      // Neither fills nor reductions should have a redop across as they
      // should have been applied an instance directly for across copies
      void record_fill(
          InstanceView* dst_view, FillView* src_view,
          const FieldMask& fill_mask, IndexSpaceExpression* expr,
          const PredEvent fill_guard, EquivalenceSet* tracing_eq,
          CopyAcrossHelper* across_helper = nullptr);
      void record_reductions(
          InstanceView* dst_view, PhysicalManager* dst_man,
          const std::list<std::pair<InstanceView*, IndexSpaceExpression*> >&
              src_views,
          const unsigned src_fidx, const unsigned dst_fidx,
          EquivalenceSet* tracing_eq,
          CopyAcrossHelper* across_helper = nullptr);
      ApEvent issue_updates(
          const PhysicalTraceInfo& trace_info, ApEvent precondition,
          const bool restricted_output = false,
          // Next args are used for across-copies
          // to indicate that the precondition already
          // describes the precondition for the
          // destination instance
          const bool manage_dst_events = true,
          std::map<InstanceView*, std::vector<ApEvent> >* dst_events = nullptr,
          int stage = -1);
    protected:
      void record_view(LogicalView* new_view);
      void resize_reductions(size_t new_size);
      void update_tracing_valid_views(
          EquivalenceSet* tracing_eq, InstanceView* src, InstanceView* dst,
          const FieldMask& mask, IndexSpaceExpression* expr,
          ReductionOpID redop) const;
      void record_instance_update(
          InstanceView* dst_view, InstanceView* src_view,
          PhysicalManager* src_man, const FieldMask& src_mask,
          IndexSpaceExpression* expr, EquivalenceSet* tracing_eq,
          ReductionOpID redop, CopyAcrossHelper* across_helper);
      struct SelectSourcesResult;
      const SelectSourcesResult& select_sources(
          InstanceView* target, PhysicalManager* target_manager,
          const std::vector<InstanceView*>& sources);
      bool perform_updates(
          const MapView<InstanceView*, op::FieldMaskMap<Update> >& updates,
          const PhysicalTraceInfo& trace_info, const ApEvent all_precondition,
          std::set<RtEvent>& recorded_events, const int redop_index,
          const bool manage_dst_events, const bool restricted_output,
          std::map<InstanceView*, std::vector<ApEvent> >* dst_events);
      void issue_fills(
          InstanceView* target, const std::vector<FillUpdate*>& fills,
          std::set<RtEvent>& recorded_events, const ApEvent precondition,
          const FieldMask& fill_mask, const PhysicalTraceInfo& trace_info,
          const bool manage_dst_events, const bool restricted_output,
          std::vector<ApEvent>* dst_events);
      void issue_copies(
          InstanceView* target,
          std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
          std::set<RtEvent>& recorded_events, const ApEvent precondition,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          const bool manage_dst_events, const bool restricted_output,
          std::vector<ApEvent>* dst_events);
    public:
      inline void clear_update_fields(void) { update_fields.clear(); }
      inline bool has_update_fields(void) const { return !!update_fields; }
      inline const FieldMask& get_update_fields(void) const
      {
        return update_fields;
      }
    public:
      static void handle_aggregation(const void* args);
    public:
      const AddressSpaceID local_space;
      PhysicalAnalysis* const analysis;
      CollectiveMapping* const collective_mapping;
      const unsigned src_index;
      const unsigned dst_index;
      const RtEvent guard_precondition;
      const PredEvent predicate_guard;
      const bool track_events;
    protected:
      FieldMask update_fields;
      op::map<InstanceView*, op::FieldMaskMap<Update> > sources;
      std::vector</*vector over reduction epochs*/
                  op::map<InstanceView*, op::FieldMaskMap<Update> > >
          reductions;
      // Figure out the reduction operator is for each epoch of a
      // given destination instance and field
      std::map<
          std::pair<InstanceView*, unsigned /*dst fidx*/>,
          std::vector<ReductionOpID> >
          reduction_epochs;
      std::set<LogicalView*> all_views;  // used for reference counting
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
        SelectSourcesResult(
            std::vector<InstanceView*>&& srcs, std::vector<unsigned>&& rank,
            std::map<unsigned, PhysicalManager*>&& pts)
          : sources(srcs), ranking(rank), points(pts)
        { }
      public:
        inline bool matches(const std::vector<InstanceView*>& srcs) const
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
        std::map<unsigned, PhysicalManager*> points;
      };
      // Cached calls to the mapper for selecting sources
      std::map<
          std::pair<InstanceView*, PhysicalManager*>,
          std::vector<SelectSourcesResult> >
          mapper_queries;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COPY_FILL_AGGREGATOR_H__
