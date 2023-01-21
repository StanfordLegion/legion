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

#ifndef __LOGGING_WRAPPER_H__
#define __LOGGING_WRAPPER_H__

#include "mappers/forwarding_mapper.h"

namespace Legion {
namespace Mapping {

/**
 * \class LoggingWrapper
 * Logs the inputs and outputs of callbacks invoked on the wrapped mapper.
 *
 * To use with your own mapper, replace any use of `new MyMapper(...)` in your
 * code with `new LoggingWrapper(new MyMapper(...))` and run with
 * `-level mapper=2`. Enabling Realm-level instance reporting might also be
 * useful (`-level inst=1`).
 *
 * If you are not already using a custom mapper, you can define something like
 * the following:
 * ```
 * static void update_mappers(Machine machine, Runtime* rt,
 *                            const std::set<Processor>& local_procs) {
 *   rt->replace_default_mapper(new LoggingWrapper(new DefaultMapper(
 *     rt->get_mapper_runtime(), machine, *(local_procs.begin()))));
 * }
 *```
 * and invoke `Runtime::add_registration_callback(update_mappers)` at some
 * point before the runtime is started, e.g. in `main` right before calling
 * `Runtime::start(...)`.
 *
 * Assumes that sharding functors are pure, and calls them an additional number
 * of times compared to normal execution.
 *
 * Currently only supports task, inline mapping and copy API calls.
 */
// TODO:
// - Include other mapping calls.
// - Provide some control over the level of detail.
class LoggingWrapper : public ForwardingMapper {
 public:
  LoggingWrapper(Mapper* mapper, Logger* logger = NULL);
  virtual ~LoggingWrapper();
#ifndef NO_LEGION_CONTROL_REPLICATION
 private:
  template<typename OPERATION>
  void select_sharding_functor_impl(const MapperContext ctx,
                                    const OPERATION& op,
                                    const SelectShardingFunctorInput& input,
                                    SelectShardingFunctorOutput& output);
 public:
  virtual void map_replicate_task(const MapperContext ctx,
                                  const Task& task,
                                  const MapTaskInput& input,
                                  const MapTaskOutput& default_output,
                                  MapReplicateTaskOutput& output);
  using ForwardingMapper::select_sharding_functor;
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Copy& copy,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);
#endif // NO_LEGION_CONTROL_REPLICATION
 public:
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
  virtual void map_copy(const MapperContext ctx,
                        const Copy& copy,
                        const MapCopyInput& input,
                        MapCopyOutput& output);
  virtual void select_copy_sources(const MapperContext ctx,
                                   const Copy& copy,
                                   const SelectCopySrcInput& input,
                                   SelectCopySrcOutput& output);
 private:
  Logger* logger;
};

}; // namespace Mapping
}; // namespace Legion

#endif // __LOGGING_WRAPPER_H___
