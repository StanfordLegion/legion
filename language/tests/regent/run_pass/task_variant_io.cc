/* Copyright 2023 Stanford University
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

#include "task_variant_io.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class VariantMapper : public DefaultMapper
{
public:
  VariantMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name);
  virtual void default_policy_rank_processor_kinds(
                                    MapperContext ctx, const Task &task, 
                                    std::vector<Processor::Kind> &ranking);
};

VariantMapper::VariantMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void VariantMapper::default_policy_rank_processor_kinds(MapperContext ctx,
                        const Task &task, std::vector<Processor::Kind> &ranking)
{
  ranking.resize(4);
  ranking[0] = Processor::TOC_PROC;
  ranking[1] = Processor::PROC_SET;
  ranking[2] = Processor::IO_PROC;
  ranking[3] = Processor::LOC_PROC;
}
static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    VariantMapper* mapper = new VariantMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "variant_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
