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

#include "mappers/logging_wrapper.h"

#include "realm/logging.h"
#include "mappers/mapping_utilities.h"

using namespace Legion::Mapping::Utilities;

Realm::Logger log_mapper("mapper");

namespace Legion {
namespace Mapping {

class MessageBuffer {
 public:
  MessageBuffer(MapperRuntime* _runtime, const MapperContext _ctx)
      : runtime(_runtime), ctx(_ctx) {
    // Do nothing;
  }
  ~MessageBuffer() {
    for (const std::stringstream& ss : lines) {
      log_mapper.info() << ss.str();
    }
  }
 public:
  std::stringstream& line() {
    lines.emplace_back();
    return lines.back();
  }
  void report(const RegionRequirement& req,
              const std::vector<PhysicalInstance>& instances,
              unsigned req_idx) {
    line() << "    " << to_string(runtime, ctx, req, req_idx);
    for (PhysicalInstance inst : instances) {
      line() << "      " << to_string(runtime, ctx, inst);
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
    for (Processor proc : output.target_procs) {
      ss << " " << proc;
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
  std::vector<std::stringstream> lines;
};

LoggingWrapper::LoggingWrapper(Mapper* mapper)
    : ForwardingMapper(mapper) {
  Machine machine = Machine::get_machine();
  AddressSpace rank = Processor::get_executing_processor().address_space();
  std::cout << "Memories on rank " << rank << ":" << std::endl;
  Machine::MemoryQuery mem_query(machine);
  mem_query.local_address_space();
  for (Memory mem : mem_query) {
    std::cout << "  " << mem << " (" << to_string(mem.kind()) << ")"
              << std::endl;
  }
  std::cout << "Processors on rank " << rank << ":" << std::endl;
  Machine::ProcessorQuery proc_query(machine);
  proc_query.local_address_space();
  for (Processor proc : proc_query) {
    std::cout << "  " << proc << " (" << to_string(proc.kind()) << ") can see";
    Machine::MemoryQuery mem_query(Machine::get_machine());
    mem_query.has_affinity_to(proc);
    for (Memory mem : mem_query) {
      Machine::AffinityDetails details;
      machine.has_affinity(proc, mem, &details);
      std::cout << " " << mem << "(bw=" << details.bandwidth << ")";
    }
    std::cout << std::endl;
  }
  std::cout.flush();
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
  buf.line() << "MAP_REPLICATE_TASK for " << to_string(runtime, ctx, task);
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
  buf.line() << "SLICE_TASK for " << to_string(runtime, ctx, task);
  buf.line() << "  INPUT: " << to_string(runtime, ctx, input.domain);
  mapper->slice_task(ctx, task, input, output);
  buf.line() << "  OUTPUT:";
  for (const TaskSlice& slice : output.slices) {
    buf.line() << "    " << to_string(runtime, ctx, slice.domain)
               << " -> " << slice.proc;
  }
}

void LoggingWrapper::map_task(const MapperContext ctx,
                              const Task& task,
                              const MapTaskInput& input,
                              MapTaskOutput& output) {
  MessageBuffer buf(runtime, ctx);
  buf.line() << "MAP_TASK for " << to_string(runtime, ctx, task);
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
  buf.line() << "SELECT_TASK_SOURCES for " << to_string(runtime, ctx, task);
  buf.line() << "  INPUT:";
  buf.report(task.regions[input.region_req_index],
             input.source_instances,
             input.region_req_index);
  mapper->select_task_sources(ctx, task, input, output);
  buf.line() << "  OUTPUT:";
  for (PhysicalInstance inst : output.chosen_ranking) {
    buf.line() << "      " << to_string(runtime, ctx, inst);
  }
}

}; // namespace Mapping
}; // namespace Legion