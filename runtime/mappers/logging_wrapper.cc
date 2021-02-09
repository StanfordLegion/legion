/* Copyright 2021 Stanford University, NVIDIA Corporation
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

#include "mappers/logging_wrapper.h"

#include "realm/logging.h"
#include "mappers/mapping_utilities.h"

using namespace Legion::Mapping::Utilities;

Realm::Logger log_maplog("mapper");

namespace Legion {
namespace Mapping {

class MessageBuffer {
 public:
  MessageBuffer(MapperRuntime* _runtime, const MapperContext _ctx)
      : runtime(_runtime), ctx(_ctx) {
    // Do nothing;
  }
  ~MessageBuffer() {
    for (std::vector<std::stringstream*>::iterator it = lines.begin();
         it != lines.end(); ++it) {
      log_maplog.info() << (*it)->str();
      delete(*it);
    }
  }
 public:
  std::stringstream& line() {
    lines.push_back(new std::stringstream());
    return *(lines.back());
  }
  void report(const RegionRequirement& req,
              const std::vector<PhysicalInstance>& instances,
              unsigned req_idx) {
    line() << "    " << to_string(runtime, ctx, req, req_idx);
    for (std::vector<PhysicalInstance>::const_iterator it = instances.begin();
         it != instances.end(); ++it) {
      line() << "      " << to_string(runtime, ctx, *it);
    }
  }
  void report(const std::vector<RegionRequirement>& reqs,
              const std::vector<std::vector<PhysicalInstance> >& instances) {
    for (size_t idx = 0; idx < reqs.size(); ++idx) {
      report(reqs[idx], instances[idx], idx);
    }
  }
  void report(const Task& task, const Mapper::MapTaskOutput& output) {
    std::stringstream& ss = line();
    ss << "  TARGET PROCS:";
    for (std::vector<Processor>::const_iterator it =
           output.target_procs.begin(); it != output.target_procs.end(); ++it) {
      ss << " " << *it;
    }
    const char *variant =
      runtime->find_task_variant_name(ctx, task.task_id, output.chosen_variant);
    line() << "  CHOSEN VARIANT: " << variant;
    line() << "  OUTPUT INSTANCES:";
    report(task.regions, output.chosen_instances);
  }
 private:
  MapperRuntime* const runtime;
  const MapperContext ctx;
  std::vector<std::stringstream*> lines;
};

LoggingWrapper::LoggingWrapper(Mapper* mapper)
    : ForwardingMapper(mapper) {
  MessageBuffer buf(runtime, NULL);
  Machine machine = Machine::get_machine();
  AddressSpace rank = Processor::get_executing_processor().address_space();
  buf.line() << "Memories on rank " << rank << ":";
  Machine::MemoryQuery mem_query(machine);
  mem_query.local_address_space();
  for (Machine::MemoryQuery::iterator it = mem_query.begin();
       it != mem_query.end(); ++it) {
    buf.line() << "  " << *it << " (" << to_string(it->kind()) << ")";
  }
  buf.line() << "Processors on rank " << rank << ":";
  Machine::ProcessorQuery proc_query(machine);
  proc_query.local_address_space();
  for (Machine::ProcessorQuery::iterator pit = proc_query.begin();
       pit != proc_query.end(); ++pit) {
    std::stringstream& line = buf.line();
    line << "  " << *pit << " (" << to_string(pit->kind()) << ") can see";
    Machine::MemoryQuery mem_query(Machine::get_machine());
    mem_query.has_affinity_to(*pit);
    for (Machine::MemoryQuery::iterator mit = mem_query.begin();
         mit != mem_query.end(); ++mit) {
      Machine::AffinityDetails details;
      machine.has_affinity(*pit, *mit, &details);
      line << " " << *mit << "(bw=" << details.bandwidth << ")";
    }
  }
}

LoggingWrapper::~LoggingWrapper() {
  // Do nothing.
}

#ifndef NO_LEGION_CONTROL_REPLICATION
void LoggingWrapper::map_replicate_task(const MapperContext ctx,
                                        const Task& task,
                                        const MapTaskInput& input,
                                        const MapTaskOutput& default_output,
                                        MapReplicateTaskOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "MAP_REPLICATE_TASK for " << to_string(runtime, ctx, task)
             << " <" << task.get_unique_id() << ">";
  buf.line() << "  INPUT:";
  buf.report(task.regions, input.valid_instances);
  mapper->map_replicate_task(ctx, task, input, default_output, output);
  buf.line() << "  OUTPUT:";
  for (unsigned i = 0; i < output.task_mappings.size(); ++i) {
    std::stringstream& ss = buf.line();
    ss << "  REPLICANT " << i;
    if (!output.control_replication_map.empty()) {
      ss << " -> " << output.control_replication_map[i];
    }
    buf.report(task, output.task_mappings[i]);
  }
}
#endif // NO_LEGION_CONTROL_REPLICATION

void LoggingWrapper::slice_task(const MapperContext ctx,
                                const Task& task,
                                const SliceTaskInput& input,
                                SliceTaskOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "SLICE_TASK for " << to_string(runtime, ctx, task)
             << " <" << task.get_unique_id() << ">";
  buf.line() << "  INPUT: " << to_string(runtime, ctx, input.domain);
  mapper->slice_task(ctx, task, input, output);
  buf.line() << "  OUTPUT:";
  for (std::vector<TaskSlice>::const_iterator it = output.slices.begin();
       it != output.slices.end(); ++it) {
    buf.line() << "    " << to_string(runtime, ctx, it->domain)
               << " -> " << it->proc;
  }
}

void LoggingWrapper::map_task(const MapperContext ctx,
                              const Task& task,
                              const MapTaskInput& input,
                              MapTaskOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "MAP_TASK for " << to_string(runtime, ctx, task)
             << " <" << task.get_unique_id() << ">";
  buf.line() << "  INPUT:";
  buf.report(task.regions, input.valid_instances);
  mapper->map_task(ctx, task, input, output);
  buf.report(task, output);
}

void LoggingWrapper::select_task_sources(const MapperContext ctx,
                                         const Task& task,
                                         const SelectTaskSrcInput& input,
                                         SelectTaskSrcOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "SELECT_TASK_SOURCES for " << to_string(runtime, ctx, task)
             << " <" << task.get_unique_id() << ">";
  buf.line() << "  INPUT:";
  buf.report(task.regions[input.region_req_index],
             input.source_instances,
             input.region_req_index);
  buf.line() << "  TARGET:";
  buf.line() << "    " << to_string(runtime, ctx, input.target);
  mapper->select_task_sources(ctx, task, input, output);
  buf.line() << "  OUTPUT:";
  for (std::deque<PhysicalInstance>::iterator it =
         output.chosen_ranking.begin();
       it != output.chosen_ranking.end(); ++it) {
    buf.line() << "      " << to_string(runtime, ctx, *it);
  }
}

void LoggingWrapper::map_inline(const MapperContext ctx,
                                const InlineMapping& inline_op,
                                const MapInlineInput& input,
                                MapInlineOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "MAP_INLINE in "
             << to_string(runtime, ctx, *(inline_op.get_parent_task()))
             << " <" << inline_op.get_unique_id() << ">";
  buf.line() << "  INPUT:";
  buf.report(inline_op.requirement, input.valid_instances, 0);
  mapper->map_inline(ctx, inline_op, input, output);
  buf.line() << "  OUTPUT:";
  buf.report(inline_op.requirement, output.chosen_instances, 0);
}

void LoggingWrapper::select_inline_sources(const MapperContext ctx,
                                           const InlineMapping& inline_op,
                                           const SelectInlineSrcInput& input,
                                           SelectInlineSrcOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "SELECT_INLINE_SOURCES in "
             << to_string(runtime, ctx, *(inline_op.get_parent_task()))
             << " <" << inline_op.get_unique_id() << ">";
  buf.line() << "  INPUT:";
  buf.report(inline_op.requirement, input.source_instances, 0);
  buf.line() << "  TARGET:";
  buf.line() << "      " << to_string(runtime, ctx, input.target);
  mapper->select_inline_sources(ctx, inline_op, input, output);
  buf.line() << "  OUTPUT:";
  for (std::deque<PhysicalInstance>::iterator it =
         output.chosen_ranking.begin();
       it != output.chosen_ranking.end(); ++it) {
    buf.line() << "      " << to_string(runtime, ctx, *it);
  }
}

}; // namespace Mapping
}; // namespace Legion
