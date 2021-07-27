/* Copyright 2021 Stanford University
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

#include "circuit_mapper.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

static LegionRuntime::Logger::Category log_circuit("circuit");

class CircuitMapper : public DefaultMapper
{
public:
  CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name,
                std::vector<Processor>* procs_list);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);
  virtual Memory default_policy_select_target_memory(
                                MapperContext ctx,
                                Processor target_proc,
                                const RegionRequirement &req,
                                MemoryConstraint mc = MemoryConstraint());
  virtual LogicalRegion default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &constraints,
                                bool force_new_instances,
                                bool meets_constraints);
  virtual void map_copy(const MapperContext ctx,
                        const Copy &copy,
                        const MapCopyInput &input,
                        MapCopyOutput &output);
  template<bool IS_SRC>
  void circuit_create_copy_instance(MapperContext ctx, const Copy &copy,
                                    const RegionRequirement &req, unsigned index,
                                    std::vector<PhysicalInstance> &instances);
private:
  std::vector<Processor>& procs_list;
};

CircuitMapper::CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list)
  : DefaultMapper(rt, machine, local, mapper_name)
  , procs_list(*_procs_list)
{
}

LogicalRegion CircuitMapper::default_policy_select_instance_region(
                              MapperContext ctx, Memory target_memory,
                              const RegionRequirement &req,
                              const LayoutConstraintSet &constraints,
                              bool force_new_instances,
                              bool meets_constraints)
{
  return req.region;
}

Processor CircuitMapper::default_policy_select_initial_processor(
                                            MapperContext ctx, const Task &task)
{
  if (same_address_space || task.is_index_space || task.index_point.is_null()) {
    return DefaultMapper::default_policy_select_initial_processor(ctx, task);
  }

  assert(task.index_point.dim == 1 && task.sharding_space.exists());
  coord_t index = task.index_point[0];
  size_t bounds = runtime->get_index_space_domain(task.sharding_space).get_volume();

  VariantInfo info =
    default_find_preferred_variant(task, ctx, false/*needs tight*/);
  switch (info.proc_kind)
  {
    case Processor::LOC_PROC:
      return remote_cpus[index * remote_cpus.size() / bounds];
    case Processor::TOC_PROC:
      return remote_gpus[index * remote_gpus.size() / bounds];
    case Processor::IO_PROC:
      return remote_ios[index * remote_ios.size() / bounds];
    case Processor::OMP_PROC:
      return remote_omps[index * remote_omps.size() / bounds];
    case Processor::PY_PROC:
      return remote_pys[index * remote_pys.size() / bounds];
    default: // make warnings go away
      break;
  }

  assert(false);
}

void CircuitMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

Memory CircuitMapper::default_policy_select_target_memory(MapperContext ctx,
                                                          Processor target_proc,
                                                          const RegionRequirement &req,
                                                          MemoryConstraint mc)
{
  if (target_proc.kind() != Processor::TOC_PROC ||
      !runtime->has_parent_logical_partition(ctx, req.region))
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req);
  LogicalRegion parent = runtime->get_parent_logical_region(ctx,
      runtime->get_parent_logical_partition(ctx, req.region));
  if (!runtime->has_parent_logical_partition(ctx, parent))
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req);
  DomainPoint color = runtime->get_logical_region_color_point(ctx, parent);
  if (color[0] > 0)
  {
    Machine::MemoryQuery visible_memories(machine);
    visible_memories.has_affinity_to(target_proc);
    visible_memories.only_kind(Memory::Z_COPY_MEM);
    assert(visible_memories.count() > 0);
    return *visible_memories.begin();
  }
  else
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
}

void CircuitMapper::map_copy(const MapperContext ctx,
                             const Copy &copy,
                             const MapCopyInput &input,
                             MapCopyOutput &output)
{
  log_circuit.spew("Circuit mapper map_copy");
  if (strcmp(copy.parent_task->get_task_name(), "toplevel") == 0)
  {
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
    {
      // Always use a virtual instance for the source.
      output.src_instances[idx].clear();
      output.src_instances[idx].push_back(
        PhysicalInstance::get_virtual_instance());

      // Place the destination instance on the remote node.
      output.dst_instances[idx].clear();
      if (!copy.dst_requirements[idx].is_restricted()) {
        // Call a customized method to create an instance on the desired node.
        circuit_create_copy_instance<false/*is src*/>(ctx, copy, 
          copy.dst_requirements[idx], idx, output.dst_instances[idx]);
      } else {
        // If it's restricted, just take the instance. This will only
        // happen inside the shard task.
        output.dst_instances[idx] = input.dst_instances[idx];
        if (!output.dst_instances[idx].empty())
          runtime->acquire_and_filter_instances(ctx,
                                  output.dst_instances[idx]);
      }
    }
  }
  else
  {
    return DefaultMapper::map_copy(ctx, copy, input, output);
  }
}

//--------------------------------------------------------------------------
template<bool IS_SRC>
void CircuitMapper::circuit_create_copy_instance(MapperContext ctx,
                     const Copy &copy, const RegionRequirement &req, 
                     unsigned idx, std::vector<PhysicalInstance> &instances)
//--------------------------------------------------------------------------
{
  // This method is identical to the default version except that it
  // chooses an intelligent memory based on the destination of the
  // copy.

  // See if we have all the fields covered
  std::set<FieldID> missing_fields = req.privilege_fields;
  for (std::vector<PhysicalInstance>::const_iterator it = 
        instances.begin(); it != instances.end(); it++)
  {
    it->remove_space_fields(missing_fields);
    if (missing_fields.empty())
      break;
  }
  if (missing_fields.empty())
    return;
  // If we still have fields, we need to make an instance
  // We clearly need to take a guess, let's see if we can find
  // one of our instances to use.

  // ELLIOTT: Get the remote node here.
  Color index = runtime->get_logical_region_color(ctx, copy.src_requirements[idx].region);
  Memory target_memory = default_policy_select_target_memory(ctx,
                           procs_list[index % procs_list.size()],
                           req);
  log_circuit.warning("Building instance for copy of a region with index %u to be in memory %llx",
                      index, target_memory.id);
  bool force_new_instances = false;
  LayoutConstraintID our_layout_id = 
   default_policy_select_layout_constraints(ctx, target_memory, 
                                            req, COPY_MAPPING,
                                            true/*needs check*/, 
                                            force_new_instances);
  LayoutConstraintSet creation_constraints = 
              runtime->find_layout_constraints(ctx, our_layout_id);
  creation_constraints.add_constraint(
      FieldConstraint(missing_fields,
                      false/*contig*/, false/*inorder*/));
  instances.resize(instances.size() + 1);
  if (!default_make_instance(ctx, target_memory, 
        creation_constraints, instances.back(), 
        COPY_MAPPING, force_new_instances, true/*meets*/, req))
  {
    // If we failed to make it that is bad
    log_circuit.error("Circuit mapper failed allocation for "
                   "%s region requirement %d of explicit "
                   "region-to-region copy operation in task %s "
                   "(ID %lld) in memory " IDFMT " for processor "
                   IDFMT ". This means the working set of your "
                   "application is too big for the allotted "
                   "capacity of the given memory under the default "
                   "mapper's mapping scheme. You have three "
                   "choices: ask Realm to allocate more memory, "
                   "write a custom mapper to better manage working "
                   "sets, or find a bigger machine. Good luck!",
                   IS_SRC ? "source" : "destination", idx, 
                   copy.parent_task->get_task_name(),
                   copy.parent_task->get_unique_id(),
		       target_memory.id,
		       copy.parent_task->current_proc.id);
    assert(false);
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();

  Machine::ProcessorQuery procs_query(machine);
  procs_query.only_kind(Processor::TOC_PROC);
  for (Machine::ProcessorQuery::iterator it = procs_query.begin();
        it != procs_query.end(); it++)
    procs_list->push_back(*it);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    CircuitMapper* mapper = new CircuitMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "circuit_mapper",
                                              procs_list);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}
