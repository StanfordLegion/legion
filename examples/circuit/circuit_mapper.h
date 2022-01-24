/* Copyright 2022 Stanford University
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

#ifndef __CIRCUIT_MAPPER_H__
#define __CIRCUIT_MAPPER_H__

#include "legion.h"
#include "default_mapper.h"
#include "circuit.h"

using namespace Legion;
using namespace Legion::Mapping;

class CircuitMapper : public DefaultMapper {
public:
  CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name,
                std::vector<Processor>* procs_list,
                std::vector<Memory>* sysmems_list,
                std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
                std::map<Processor, Memory>* proc_sysmems,
                std::map<Processor, Memory>* proc_fbmems,
                std::map<Processor, Memory>* proc_zcmems);
public:
  virtual void map_task(const MapperContext      ctx,
                        const Task&              task,
                        const MapTaskInput&      input,
                              MapTaskOutput&     output);
  virtual void map_inline(const MapperContext    ctx,
                          const InlineMapping&   inline_op,
                          const MapInlineInput&  input,
                                MapInlineOutput& output);
protected:
  void map_circuit_region(const MapperContext ctx, LogicalRegion region,
                          Processor target_proc, Memory target,
                          std::vector<PhysicalInstance> &instanes,
                          const std::set<FieldID> &privilege_fields,
                          ReductionOpID redop);
protected:
  std::vector<Processor>& procs_list;
  std::vector<Memory>& sysmems_list;
  std::map<Memory, std::vector<Processor> >& sysmem_local_procs;
  std::map<Processor, Memory>& proc_sysmems;
  std::map<Processor, Memory>& proc_fbmems;
  std::map<Processor, Memory>& proc_zcmems;
protected:
  // For memoizing mapping instances
  std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance> local_instances;
  std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance> reduction_instances;
};

void update_mappers(Machine machine, Runtime *rt,
                    const std::set<Processor> &local_procs);
#endif // __CIRCUIT_MAPPER_H__
