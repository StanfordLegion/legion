#include "miniaero_mapper.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

#define SPMD_SHARD_USE_IO_PROC 1

LegionRuntime::Logger::Category log_mapper("mapper");

class MiniAeroMapper : public DefaultMapper
{
public:
  MiniAeroMapper(MapperRuntime *rt, Machine machine, Processor local,
                 const char *mapper_name);

  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);
  virtual LogicalRegion default_policy_select_instance_region(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    const LayoutConstraintSet &constraints,
                                    bool force_new_instances,
                                    bool meets_constraints);
};

MiniAeroMapper::MiniAeroMapper(MapperRuntime *rt, Machine machine, Processor local,
                               const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void MiniAeroMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

LogicalRegion MiniAeroMapper::default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &layout_constraints,
                                bool force_new_instances,
                                bool meets_constraints)
{
  return req.region;
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    MiniAeroMapper* mapper = new MiniAeroMapper(runtime->get_mapper_runtime(),
                                                machine, *it, "miniaero_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}
