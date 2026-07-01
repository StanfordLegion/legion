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

#ifndef __LEGION_PHYSICAL_TRACE_H__
#define __LEGION_PHYSICAL_TRACE_H__

#include "legion/tracing/logical.h"
#include "legion/utilities/coordinates.h"

namespace Legion {
  namespace Internal {

    enum ReplayableStatus {
      REPLAYABLE,
      NOT_REPLAYABLE_BLOCKING,
      NOT_REPLAYABLE_CONSENSUS,
      NOT_REPLAYABLE_VIRTUAL,
      NOT_REPLAYABLE_REMOTE_SHARD,
      NOT_REPLAYABLE_NON_LEAF,
      NOT_REPLAYABLE_VARIABLE_RETURN,
    };

    enum IdempotencyStatus {
      IDEMPOTENT,
      NOT_IDEMPOTENT_SUBSUMPTION,
      NOT_IDEMPOTENT_ANTIDEPENDENT,
      NOT_IDEMPOTENT_REMOTE_SHARD,
    };

    /**
     * \class PhysicalTrace
     * This class is used for memoizing the dynamic physical dependence
     * analysis for series of operations in a given task's context.
     */
    class PhysicalTrace {
    public:
      PhysicalTrace(LogicalTrace* logical_trace);
      PhysicalTrace(const PhysicalTrace& rhs) = delete;
      ~PhysicalTrace(void);
    public:
      PhysicalTrace& operator=(const PhysicalTrace& rhs) = delete;
    public:
      inline bool has_current_template(void) const
      {
        return (current_template != nullptr);
      }
      inline PhysicalTemplate* get_current_template(void) const
      {
        return current_template;
      }
      inline const std::vector<Processor>& get_replay_targets(void)
      {
        return replay_targets;
      }
      inline bool is_recording(void) const { return recording; }
      inline bool is_replaying(void) const { return !recording; }
      inline bool is_recurrent(void) const { return recurrent; }
    public:
      void record_parent_req_fields(unsigned index, const FieldMask& mask);
      void find_condition_sets(std::map<EquivalenceSet*, unsigned>& sets) const;
      void refresh_condition_sets(
          FenceOp* op, std::set<RtEvent>& refresh_ready) const;
      bool begin_physical_trace(
          BeginOp* op, std::set<RtEvent>& map_applied_conditions,
          std::set<ApEvent>& execution_preconditions);
      void complete_physical_trace(
          CompleteOp* op, std::set<RtEvent>& map_applied_conditions,
          std::set<ApEvent>& execution_preconditions, bool has_blocking_call);
      bool replay_physical_trace(
          RecurrentOp* op, std::set<RtEvent>& map_applied_events,
          std::set<ApEvent>& execution_preconditions, bool has_blocking_call,
          bool has_intermediate_fence);
      void invalidate_equivalence_sets(void) const;
    protected:
      bool find_replay_template(
          BeginOp* op, std::set<RtEvent>& map_applied_conditions,
          std::set<ApEvent>& execution_preconditions);
      void begin_replay(
          BeginOp* op, bool recurrent, bool has_intermediate_fence);
      bool complete_recording(
          CompleteOp* op, std::set<RtEvent>& map_applied_conditions,
          std::set<ApEvent>& execution_preconditions, bool has_blocking_call);
      void complete_replay(std::set<ApEvent>& completion_events);
    public:
      const LogicalTrace* const logical_trace;
      const bool perform_fence_elision;
    private:
      mutable LocalLock trace_lock;
      // This is a mapping from the parent region requirements
      // to the sets of fields referred to in the trace. We use
      // this to find the equivalence sets for a template
      ctx::map<unsigned, FieldMask> parent_req_fields;
      std::vector<PhysicalTemplate*> templates;
      PhysicalTemplate* current_template;
      unsigned nonreplayable_count;
      unsigned new_template_count;
    private:
      std::vector<Processor> replay_targets;
      bool recording;
      bool recurrent;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/tracing/physical.inl"

#endif  // __LEGION_PHYSICAL_TRACE_H__
