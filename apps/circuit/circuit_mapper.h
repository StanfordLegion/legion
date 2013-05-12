/* Copyright 2013 Stanford University
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


#ifndef __CIRCUIT_MAPPER__
#define __CIRCUIT_MAPPER__

#include "legion.h"
#include "default_mapper.h"

class CircuitMapper : public DefaultMapper {
public:
  CircuitMapper(Machine *m, HighLevelRuntime *rt, Processor local);
public:
  virtual bool spawn_task(const Task *task);
  virtual Processor select_target_processor(const Task *task);
  virtual Processor target_task_steal(const std::set<Processor> &blacklisted);
  virtual void permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal);
  virtual bool map_task_region(const Task *task, Processor target, 
                                MappingTagID tag, bool inline_mapping,
                                const RegionRequirement &req, unsigned index,
                                const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                std::vector<Memory> &target_ranking,
                                std::set<FieldID> &additional_fields,
                                bool &enable_WAR_optimization);
  virtual void rank_copy_target(const Task *task, Processor target,
                                MappingTagID tag, bool inline_mapping,
                                const RegionRequirement &req, unsigned index,
                                const std::set<Memory> &current_instances,
                                std::set<Memory> &to_reuse,
                                std::vector<Memory> &to_create,
                                bool &create_one);
  virtual void slice_index_space(const Task *task, const IndexSpace &index_space,
                                  std::vector<Mapper::DomainSplit> &slices);
public:
  std::vector<Processor> cpu_procs;
  std::vector<Processor> gpu_procs;
  Memory gasnet_mem;
  Memory zero_copy_mem;
  Memory framebuffer_mem;
};

#endif // __CIRCUIT_MAPPER__

