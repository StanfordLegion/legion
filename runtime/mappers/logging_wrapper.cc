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

#include "mappers/logging_wrapper.h"

#include "realm/logging.h"
#include "mappers/mapping_utilities.h"

using namespace Legion::Mapping::Utilities;

Legion::Logger log_maplog("mapper");

namespace Legion {
namespace Mapping {

class MessageBuffer {
 public:
  MessageBuffer(MapperRuntime* _runtime,
                const MapperContext _ctx,
                Legion::Logger* _logger)
      : runtime(_runtime), ctx(_ctx), logger(_logger) {
    // Do nothing;
  }
  ~MessageBuffer() {
    for (std::vector<std::stringstream*>::iterator it = lines.begin();
         it != lines.end(); ++it) {
      logger->info() << (*it)->str();
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
  void report(const RegionRequirement& req,
              const PhysicalInstance& inst,
              unsigned req_idx) {
    line() << "    " << to_string(runtime, ctx, req, req_idx);
    line() << "      " << to_string(runtime, ctx, inst);
  }
  void report(const std::vector<RegionRequirement>& reqs,
              const std::vector<std::vector<PhysicalInstance> >& instances) {
    for (size_t idx = 0; idx < reqs.size(); ++idx) {
      report(reqs[idx], instances[idx], idx);
    }
  }
  void report(const std::vector<RegionRequirement>& reqs,
              const std::vector<PhysicalInstance>& instances) {
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
    line() << "  CHOSEN INSTANCES:";
    report(task.regions, output.chosen_instances);
  }
 private:
  MapperRuntime* const runtime;
  const MapperContext ctx;
  Logger* logger;
  std::vector<std::stringstream*> lines;
};

LoggingWrapper::LoggingWrapper(Mapper* mapper, Logger* _logger)
    : ForwardingMapper(mapper),
      logger(_logger != NULL ? _logger : &log_maplog) {
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, NULL, logger);
  Machine machine = Machine::get_machine();
  AddressSpace rank = Processor::get_executing_processor().address_space();
  buf.line() << "Memories on rank " << rank << ":";
  Machine::MemoryQuery mem_query(machine);
  mem_query.local_address_space();
  for (Machine::MemoryQuery::iterator it = mem_query.begin();
       it != mem_query.end(); ++it) {
    buf.line() << "  " << *it << " (" << to_string(it->kind()) << "): "
               << it->capacity() << " bytes";
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
template <typename OPERATION>
void LoggingWrapper::select_sharding_functor_impl(
                              const MapperContext ctx,
                              const OPERATION& op,
                              const SelectShardingFunctorInput& input,
                              SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, op, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "SELECT_SHARDING_FUNCTOR for "
             << to_string(runtime, ctx, op, false /*include_index_point*/);
  ShardingFunctor* functor =
    Runtime::get_sharding_functor(output.chosen_functor);
  Domain point_space =
    op.is_index_space
    ? op.index_domain
    : Domain(op.index_point, op.index_point);
  Domain sharding_space =
    (op.sharding_space != IndexSpace::NO_SPACE)
    ? runtime->get_index_space_domain(ctx, op.sharding_space)
    : point_space;
  size_t num_shards = input.shard_mapping.size();
  std::vector<std::vector<DomainPoint>> points_per_shard(num_shards);
  for (Domain::DomainPointIterator it(point_space); it; ++it) {
    ShardID shard = functor->shard(*it, sharding_space, num_shards);
    points_per_shard[shard].push_back(*it);
  }
  for (size_t shard = 0; shard < num_shards; ++shard) {
    std::stringstream& ss = buf.line();
    ss << "  " << shard << " <-";
    for (const DomainPoint& p : points_per_shard[shard]) {
      ss << " " << p;
    }
  }
}

void LoggingWrapper::map_replicate_task(const MapperContext ctx,
                                        const Task& task,
                                        const MapTaskInput& input,
                                        const MapTaskOutput& default_output,
                                        MapReplicateTaskOutput& output) {
  mapper->map_replicate_task(ctx, task, input, default_output, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "MAP_REPLICATE_TASK for "
             << to_string(runtime, ctx, task, false /*include_index_point*/);
  for (unsigned i = 0; i < output.task_mappings.size(); ++i) {
    std::stringstream& ss = buf.line();
    ss << "  REPLICANT " << i;
    if (!output.control_replication_map.empty()) {
      ss << " -> " << output.control_replication_map[i];
    }
    buf.report(task, output.task_mappings[i]);
  }
}

void LoggingWrapper::select_sharding_functor(
                              const MapperContext ctx,
                              const Task& task,
                              const SelectShardingFunctorInput& input,
                              SelectShardingFunctorOutput& output) {
  select_sharding_functor_impl(ctx, task, input, output);
}

void LoggingWrapper::select_sharding_functor(
                              const MapperContext ctx,
                              const Copy& copy,
                              const SelectShardingFunctorInput& input,
                              SelectShardingFunctorOutput& output) {
  select_sharding_functor_impl(ctx, copy, input, output);
}
#endif // NO_LEGION_CONTROL_REPLICATION

void LoggingWrapper::slice_task(const MapperContext ctx,
                                const Task& task,
                                const SliceTaskInput& input,
                                SliceTaskOutput& output) {
  mapper->slice_task(ctx, task, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "SLICE_TASK for "
             << to_string(runtime, ctx, task, false /*include_index_point*/);
  for (std::vector<TaskSlice>::const_iterator it = output.slices.begin();
       it != output.slices.end(); ++it) {
    buf.line() << "  " << to_string(runtime, ctx, it->domain)
               << " -> " << it->proc;
  }
}

void LoggingWrapper::map_task(const MapperContext ctx,
                              const Task& task,
                              const MapTaskInput& input,
                              MapTaskOutput& output) {
  mapper->map_task(ctx, task, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "MAP_TASK for " << to_string(runtime, ctx, task);
  buf.report(task, output);
}

void LoggingWrapper::select_task_sources(const MapperContext ctx,
                                         const Task& task,
                                         const SelectTaskSrcInput& input,
                                         SelectTaskSrcOutput& output) {
  mapper->select_task_sources(ctx, task, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "SELECT_TASK_SOURCES for " << to_string(runtime, ctx, task);
  buf.line() << "  TARGET:";
  buf.line() << "    " << to_string(runtime, ctx,
                                    task.regions[input.region_req_index],
                                    input.region_req_index);
  buf.line() << "    " << to_string(runtime, ctx, input.target);
  buf.line() << "  SOURCES:";
  for (std::deque<PhysicalInstance>::iterator it =
         output.chosen_ranking.begin();
       it != output.chosen_ranking.end(); ++it) {
    buf.line() << "    " << to_string(runtime, ctx, *it);
  }
}

void LoggingWrapper::map_inline(const MapperContext ctx,
                                const InlineMapping& inline_op,
                                const MapInlineInput& input,
                                MapInlineOutput& output) {
  mapper->map_inline(ctx, inline_op, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "MAP_INLINE for "
             << to_string(runtime, ctx, inline_op)
             << " in "
             << to_string(runtime, ctx, *(inline_op.get_parent_task()));
  buf.report(inline_op.requirement, output.chosen_instances, 0);
}

void LoggingWrapper::select_inline_sources(const MapperContext ctx,
                                           const InlineMapping& inline_op,
                                           const SelectInlineSrcInput& input,
                                           SelectInlineSrcOutput& output) {
  mapper->select_inline_sources(ctx, inline_op, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "SELECT_INLINE_SOURCES for "
             << to_string(runtime, ctx, inline_op)
             << " in "
             << to_string(runtime, ctx, *(inline_op.get_parent_task()));
  buf.line() << "  TARGET:";
  buf.line() << "    " << to_string(runtime, ctx, inline_op.requirement, 0);
  buf.line() << "    " << to_string(runtime, ctx, input.target);
  buf.line() << "  SOURCES:";
  for (std::deque<PhysicalInstance>::iterator it =
         output.chosen_ranking.begin();
       it != output.chosen_ranking.end(); ++it) {
    buf.line() << "    " << to_string(runtime, ctx, *it);
  }
}

void LoggingWrapper::map_copy(const MapperContext ctx,
                              const Copy& copy,
                              const MapCopyInput& input,
                              MapCopyOutput& output) {
  mapper->map_copy(ctx, copy, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "MAP_COPY for " << to_string(runtime, ctx, copy);
  buf.line() << "  SRC:";
  buf.report(copy.src_requirements, output.src_instances);
  buf.line() << "  SRC_INDIRECT:";
  buf.report(copy.src_indirect_requirements, output.src_indirect_instances);
  buf.line() << "  DST_INDIRECT:";
  buf.report(copy.dst_indirect_requirements, output.dst_indirect_instances);
  buf.line() << "  DST:";
  buf.report(copy.dst_requirements, output.dst_instances);
}

void LoggingWrapper::select_copy_sources(const MapperContext ctx,
                                         const Copy& copy,
                                         const SelectCopySrcInput& input,
                                         SelectCopySrcOutput& output) {
  mapper->select_copy_sources(ctx, copy, input, output);
  if (!logger->want_info()) return;
  MessageBuffer buf(runtime, ctx, logger);
  buf.line() << "SELECT_COPY_SOURCES for " << to_string(runtime, ctx, copy)
             << " "
             << (input.is_src ? "SRC" : input.is_dst ? "DST" :
                 input.is_src_indirect ? "SRC_INDIRECT" : "DST_INDIRECT");
  const std::vector<RegionRequirement>& reqs =
    input.is_src ? copy.src_requirements :
    input.is_dst ? copy.dst_requirements :
    input.is_src_indirect ? copy.src_indirect_requirements :
    /* input.is_dst_indirect */ copy.dst_indirect_requirements;
  buf.line() << "  TARGET:";
  buf.line() << "    " << to_string(runtime, ctx,
                                    reqs[input.region_req_index],
                                    input.region_req_index);
  buf.line() << "    " << to_string(runtime, ctx, input.target);
  buf.line() << "  SOURCES:";
  for (std::deque<PhysicalInstance>::iterator it =
         output.chosen_ranking.begin();
       it != output.chosen_ranking.end(); ++it) {
    buf.line() << "    " << to_string(runtime, ctx, *it);
  }
}

}; // namespace Mapping
}; // namespace Legion
