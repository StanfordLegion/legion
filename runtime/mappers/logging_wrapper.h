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

#ifndef __LOGGING_WRAPPER_H__
#define __LOGGING_WRAPPER_H__

#include "mappers/forwarding_mapper.h"

namespace Legion {
namespace Mapping {

/**
 * \class LoggingWrapper
 * Adds logging to a wrapped mapper (use -level mapper=X).
 *
 * Currently only supports task and inline mapping API calls.
 */
// TODO:
// - Include other mapping calls.
// - Provide some control over the level of detail.
class LoggingWrapper : public ForwardingMapper {
 public:
  LoggingWrapper(Mapper* mapper);
  virtual ~LoggingWrapper();
 public:
#ifndef NO_LEGION_CONTROL_REPLICATION
  virtual void map_replicate_task(const MapperContext ctx,
                                  const Task& task,
                                  const MapTaskInput& input,
                                  const MapTaskOutput& default_output,
                                  MapReplicateTaskOutput& output);
#endif // NO_LEGION_CONTROL_REPLICATION
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
  virtual void select_task_sources(const MapperContext ctx,
                                   const Task& task,
                                   const SelectTaskSrcInput& input,
                                   SelectTaskSrcOutput& output);
  virtual void map_inline(const MapperContext ctx,
                          const InlineMapping& inline_op,
                          const MapInlineInput& input,
                          MapInlineOutput& output);
  virtual void select_inline_sources(const MapperContext ctx,
                                     const InlineMapping& inline_op,
                                     const SelectInlineSrcInput& input,
                                     SelectInlineSrcOutput& output);
};

}; // namespace Mapping
}; // namespace Legion

#endif // __LOGGING_WRAPPER_H___
