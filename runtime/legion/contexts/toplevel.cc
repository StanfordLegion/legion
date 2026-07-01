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

#include "legion/contexts/toplevel.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Top Level Context
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(
        Processor p, coord_t normal_id, coord_t implicit_id, DistributedID id,
        CollectiveMapping* mapping)
      : HeapifyMixin<TopLevelContext, InnerContext, CONTEXT_LIFETIME>(
            configure_toplevel_context(), (SingleTask*)nullptr, -1,
            false /*full inner*/, dummy_requirements, dummy_output_requirements,
            dummy_indexes, dummy_mapped, 0 /*priority*/, ApEvent::NO_AP_EVENT,
            id, false, false, false, mapping),
        root_uid(runtime->get_unique_operation_id())
    //--------------------------------------------------------------------------
    {
      legion_assert(p.exists());
      set_executing_processor(p);
      // This coordinate represents the name of the unique top-level task
      // launched by this instance of the Legion runtime
      context_coordinates.emplace_back(ContextCoordinate(
          0 /*context index*/, DomainPoint(Point<2>(normal_id, implicit_id))));
    }

    //--------------------------------------------------------------------------
    /*static*/ Mapper::ContextConfigOutput
        TopLevelContext::configure_toplevel_context(void)
    //--------------------------------------------------------------------------
    {
      // We don't need to consult with a mapper to configure the top-level
      // context because there's always going to be exactly one task in this
      // context which is the top-level task
      Mapper::ContextConfigOutput configuration;
      configuration.max_window_size = runtime->initial_task_window_size;
      configuration.hysteresis_percentage =
          runtime->initial_task_window_hysteresis;
      configuration.max_outstanding_frames = 0;
      configuration.min_tasks_to_schedule = runtime->initial_tasks_to_schedule;
      configuration.min_frames_to_schedule = 0;
      configuration.meta_task_vector_width =
          runtime->initial_meta_task_vector_width;
      configuration.max_templates_per_trace =
          LEGION_DEFAULT_MAX_TEMPLATES_PER_TRACE;
      configuration.mutable_priority = false;
      configuration.auto_tracing_enabled = false;
      configuration.auto_tracing_window_size = 0;
      configuration.auto_tracing_ruler_function = 0;
      configuration.auto_tracing_min_trace_length = 0;
      configuration.auto_tracing_max_trace_length = 0;
      configuration.auto_tracing_visit_threshold = 0;
      return configuration;
    }

    //--------------------------------------------------------------------------
    TopLevelContext::~TopLevelContext(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TopLevelContext::pack_remote_context(
        Serializer& rez, AddressSpaceID target, bool replicate)
    //--------------------------------------------------------------------------
    {
      rez.serialize(depth);
    }

    //--------------------------------------------------------------------------
    InnerContext* TopLevelContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      return nullptr;
    }

    //--------------------------------------------------------------------------
    void TopLevelContext::receive_created_region_contexts(
        const std::vector<RegionNode*>& created_nodes,
        const std::vector<EqKDTree*>& created_trees,
        std::set<RtEvent>& applied_events, const ShardMapping* mapping,
        ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent TopLevelContext::compute_equivalence_sets(
        unsigned req_index, const std::vector<EqSetTracker*>& targets,
        const std::vector<AddressSpaceID>& target_spaces,
        AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
        const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent TopLevelContext::record_output_equivalence_set(
        EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
        EquivalenceSet* expr, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    InnerContext* TopLevelContext::find_top_context(InnerContext* previous)
    //--------------------------------------------------------------------------
    {
      legion_assert(previous != nullptr);
      return previous;
    }

  }  // namespace Internal
}  // namespace Legion
