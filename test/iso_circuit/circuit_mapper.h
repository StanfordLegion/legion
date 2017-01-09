/* Copyright 2016 Stanford University
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
#include "shim_mapper.h"
#include "circuit.h"

using namespace LegionRuntime::HighLevel;

class CircuitMapper : public ShimMapper {
public:
  CircuitMapper(Machine machine, HighLevelRuntime *runtime, Processor local);
public:
  virtual void select_task_options(Task *task);
  virtual void slice_domain(const Task *task, const Domain &domain,
                            std::vector<DomainSplit> &slices);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);

  virtual void notify_mapping_failed(const Mappable *mappable);
#if 0
  virtual bool rank_copy_targets(const Mappable *mappble,
                                 const std::set<Memory> &current_instances,
                                 bool complete,
                                 size_t max_blocking_factor,
                                 std::set<Memory> &to_reuse,
                                 std::vector<Memory> &to_create,
                                 bool &create_one,
                                 size_t &blocking_factor);
#endif
protected:
  bool map_to_gpus, first;
  std::vector<Processor> all_cpus;
  std::vector<Processor> all_gpus;
  std::map<Processor, Memory> all_sysmems;
};

#endif // __CIRCUIT_MAPPER_H__
