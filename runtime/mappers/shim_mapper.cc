/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "shim_mapper.h"

namespace LegionRuntime {
  namespace HighLevel {

    // In this version the ShimMapper inherits all of its
    // functionality from DefaultMapper.

    ShimMapper::ShimMapper(Machine machine, HighLevelRuntime *rt, Processor local)
      : DefaultMapper(machine, rt, local) {}
    ShimMapper::ShimMapper(const ShimMapper &rhs)
      : DefaultMapper(rhs) {}

    ShimMapper::~ShimMapper() {}

    ShimMapper& ShimMapper::operator=(const ShimMapper &rhs)
    {
      // should never be called
      assert(false);
      return *this;
    }

    void ShimMapper::select_task_options(Task *task)
    {
      DefaultMapper::select_task_options(task);
    }

    void ShimMapper::slice_domain(const Task *task, const Domain &domain,
                      std::vector<DomainSplit> &slices)
    {
      DefaultMapper::slice_domain(task, domain, slices);
    }

    void ShimMapper::select_task_variant(Task *task)
    {
      DefaultMapper::select_task_variant(task);
    }

    bool ShimMapper::map_task(Task *task)
    {
      return DefaultMapper::map_task(task);
    }

    bool ShimMapper::map_copy(Copy *copy)
    {
      return DefaultMapper::map_copy(copy);
    }

    bool ShimMapper::map_inline(Inline *inline_operation)
    {
      return DefaultMapper::map_inline(inline_operation);
    }

    bool ShimMapper::map_must_epoch(const std::vector<Task*> &tasks,
                                    const std::vector<MappingConstraint> &constraints,
                                    MappingTagID tag)
    {
      return DefaultMapper::map_must_epoch(tasks, constraints, tag);
    }

    void ShimMapper::notify_mapping_result(const Mappable *mappable)
    {
      DefaultMapper::notify_mapping_result(mappable);
    }

    void ShimMapper::notify_mapping_failed(const Mappable *mappable)
    {
      DefaultMapper::notify_mapping_failed(mappable);
    }
  }
}
