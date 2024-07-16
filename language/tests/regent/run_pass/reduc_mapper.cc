/* Copyright 2024 Stanford University
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

#include "reduc_mapper.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class ReducMapper : public DefaultMapper
{
public:
  ReducMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name);
  bool default_policy_select_reduc_instance_reuse(const MapperContext ctx) override;
};

ReducMapper::ReducMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

bool ReducMapper::default_policy_select_reduc_instance_reuse(const MapperContext ctx)
{
  return false;
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    ReducMapper* mapper = new ReducMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "reduc_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
