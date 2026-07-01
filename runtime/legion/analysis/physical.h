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

#ifndef __LEGION_PHYSICAL_ANALYSIS_H__
#define __LEGION_PHYSICAL_ANALYSIS_H__

#include "legion/kernel/metatask.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/api/types.h"
#include "legion/utilities/fieldmask_map.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PhysicalAnalysis
     * This is the virtual base class for handling all traversals over
     * equivalence set trees to perform physical analyses
     */
    class PhysicalAnalysis : public Collectable,
                             public LocalLock {
    public:
      struct DeferPerformTraversalArgs
        : public LgTaskArgs<DeferPerformTraversalArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PERFORM_TRAVERSAL_TASK_ID;
      public:
        DeferPerformTraversalArgs(void) = default;
        DeferPerformTraversalArgs(
            PhysicalAnalysis* ana, const VersionInfo& info);
        void execute(void) const;
      public:
        PhysicalAnalysis* analysis;
        const VersionInfo* version_info;
        RtUserEvent done_event;
      };
      struct DeferPerformAnalysisArgs
        : public LgTaskArgs<DeferPerformAnalysisArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PERFORM_ANALYSIS_TASK_ID;
      public:
        DeferPerformAnalysisArgs(void) = default;
        DeferPerformAnalysisArgs(
            PhysicalAnalysis* ana, EquivalenceSet* set, const FieldMask& mask,
            RtUserEvent done, bool already_deferred = true);
        void execute(void) const;
      public:
        PhysicalAnalysis* analysis;
        EquivalenceSet* set;
        HeapifyBox<FieldMask, OPERATION_LIFETIME>* mask;
        RtUserEvent done_event;
        bool already_deferred;
      };
      struct DeferPerformRemoteArgs
        : public LgTaskArgs<DeferPerformRemoteArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PERFORM_REMOTE_TASK_ID;
      public:
        DeferPerformRemoteArgs(void) = default;
        DeferPerformRemoteArgs(PhysicalAnalysis* ana);
        void execute(void) const;
      public:
        PhysicalAnalysis* analysis;
        RtUserEvent done_event;
      };
      struct DeferPerformUpdateArgs
        : public LgTaskArgs<DeferPerformUpdateArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PERFORM_UPDATE_TASK_ID;
      public:
        DeferPerformUpdateArgs(void) = default;
        DeferPerformUpdateArgs(PhysicalAnalysis* ana);
        void execute(void) const;
      public:
        PhysicalAnalysis* analysis;
        RtUserEvent done_event;
      };
      struct DeferPerformRegistrationArgs
        : public LgTaskArgs<DeferPerformRegistrationArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_DEFER_PERFORM_REGISTRATION_TASK_ID;
      public:
        DeferPerformRegistrationArgs(void) = default;
        DeferPerformRegistrationArgs(
            PhysicalAnalysis* ana, const RegionUsage& use,
            const PhysicalTraceInfo& trace_info, ApEvent init_precondition,
            ApEvent termination, bool symbolic);
        void execute(void) const;
      public:
        PhysicalAnalysis* analysis;
        RegionUsage usage;
        const PhysicalTraceInfo* trace_info;
        ApEvent precondition;
        ApEvent termination;
        ApUserEvent instances_ready;
        RtUserEvent done_event;
        bool symbolic;
      };
      struct DeferPerformOutputArgs
        : public LgTaskArgs<DeferPerformOutputArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PERFORM_OUTPUT_TASK_ID;
      public:
        DeferPerformOutputArgs(void) = default;
        DeferPerformOutputArgs(
            PhysicalAnalysis* ana, bool track,
            const PhysicalTraceInfo& trace_info);
        void execute(void) const;
      public:
        PhysicalAnalysis* analysis;
        const PhysicalTraceInfo* trace_info;
        ApUserEvent effects_event;
      };
    public:
      // Local physical analysis
      PhysicalAnalysis(
          Operation* op, unsigned index, IndexSpaceExpression* expr,
          bool on_heap, bool immutable, bool exclusive = false,
          CollectiveMapping* mapping = nullptr, bool first_local = true);
      // Remote physical analysis
      PhysicalAnalysis(
          AddressSpaceID source, AddressSpaceID prev, Operation* op,
          unsigned index, IndexSpaceExpression* expr, bool on_heap,
          bool immutable = false, CollectiveMapping* mapping = nullptr,
          bool exclusive = false, bool first_local = true);
      PhysicalAnalysis(const PhysicalAnalysis& rhs) = delete;
      virtual ~PhysicalAnalysis(void);
    public:
      inline bool has_remote_sets(void) const { return !remote_sets.empty(); }
      inline void record_parallel_traversals(void)
      {
        parallel_traversals = true;
      }
      inline bool is_replicated(void) const
      {
        return (collective_mapping != nullptr);
      }
      inline CollectiveMapping* get_replicated_mapping(void) const
      {
        return collective_mapping;
      }
      inline bool is_collective_first_local(void) const
      {
        return collective_first_local;
      }
    public:
      void analyze(
          EquivalenceSet* set, const FieldMask& mask,
          std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
          RtEvent precondition = RtEvent::NO_RT_EVENT,
          const bool already_deferred = false);
      RtEvent defer_traversal(
          RtEvent precondition, const VersionInfo& info,
          std::set<RtEvent>& applied_events);
      void defer_analysis(
          RtEvent precondition, EquivalenceSet* set, const FieldMask& mask,
          std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
          RtUserEvent deferral_event = RtUserEvent::NO_RT_USER_EVENT,
          const bool already_deferred = true);
      RtEvent defer_remote(RtEvent precondition, std::set<RtEvent>& applied);
      RtEvent defer_updates(RtEvent precondition, std::set<RtEvent>& applied);
      RtEvent defer_registration(
          RtEvent precondition, const RegionUsage& usage,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, ApEvent init_precondition,
          ApEvent termination_event, ApEvent& instances_ready, bool symbolic);
      ApEvent defer_output(
          RtEvent precondition, const PhysicalTraceInfo& info, bool track,
          std::set<RtEvent>& applied_events);
      void record_deferred_applied_events(std::set<RtEvent>& applied);
    public:
      virtual RtEvent perform_traversal(
          RtEvent precondition, const VersionInfo& version_info,
          std::set<RtEvent>& applied_events);
      virtual bool perform_analysis(
          EquivalenceSet* set, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false) = 0;
      virtual RtEvent perform_remote(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      virtual RtEvent perform_updates(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      virtual RtEvent perform_registration(
          RtEvent precondition, const RegionUsage& usage,
          std::set<RtEvent>& applied_events, ApEvent init_precondition,
          ApEvent termination_event, ApEvent& instances_ready,
          bool symbolic = false);
      virtual ApEvent perform_output(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      virtual IndexSpace get_collective_match_space(void) const
      {
        std::abort();
      }
    public:
      void process_remote_instances(
          Deserializer& derez, std::set<RtEvent>& ready_events);
      void process_local_instances(
          const FieldMapView<LogicalView>& views, const bool local_restricted);
      void filter_remote_expressions(
          op::FieldMaskMap<IndexSpaceExpression>& exprs);
      // Return true if any are restricted
      bool report_instances(op::FieldMaskMap<LogicalView>& instances);
    public:
      // Lock taken by these methods if needed
      void record_remote(
          EquivalenceSet* set, const FieldMask& mask,
          const AddressSpaceID owner);
    public:
      // Lock must be held from caller
      void record_instance(LogicalView* view, const FieldMask& mask);
      inline void record_restriction(void) { restricted = true; }
    public:
      const AddressSpaceID previous;
      const AddressSpaceID original_source;
      IndexSpaceExpression* const analysis_expr;
    public:
      Operation* const op;
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
      op::map<AddressSpaceID, op::FieldMaskMap<EquivalenceSet> > remote_sets;
      op::FieldMaskMap<LogicalView>* recorded_instances;
    protected:
      CollectiveMapping* collective_mapping;
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

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PHYSICAL_ANALYSIS_H__
