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


#ifndef __SHIM_MAPPER_H__
#define __SHIM_MAPPER_H__

#include "legion.h"
#include "mapping_utilities.h"
#include "default_mapper.h"
#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace LegionRuntime {
  namespace HighLevel {

    /**
     * \class ShimMapper
     * The ShimMapper class provides forwards compatibility for
     * existing mappers. By using this class, mappers can guarrantee
     * that they will continue to function when the mapper API is
     * upgraded.
     */
    class ShimMapper : public DefaultMapper {
    public:
      ShimMapper(Machine machine, HighLevelRuntime *rt, Processor local);
      ShimMapper(const ShimMapper &rhs);
      virtual ~ShimMapper(void);
    public:
      ShimMapper& operator=(const ShimMapper &rhs);
    public:
      virtual void select_task_options(Task *task);
      virtual void slice_domain(const Task *task, const Domain &domain,
                                std::vector<DomainSplit> &slices);
      virtual void select_task_variant(Task *task);
      virtual bool map_task(Task *task);
      virtual bool map_copy(Copy *copy);
      virtual bool map_inline(Inline *inline_operation);
      virtual bool map_must_epoch(const std::vector<Task*> &tasks,
                                  const std::vector<MappingConstraint> &constraints,
                                  MappingTagID tag);
      virtual void notify_mapping_result(const Mappable *mappable);
      virtual void notify_mapping_failed(const Mappable *mappable);
    };

  };
};

#endif // __SHIM_MAPPER_H__

// EOF

