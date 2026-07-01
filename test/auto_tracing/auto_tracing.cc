/* Copyright 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sys/wait.h>
#include <unistd.h>
#include <optional>
#include <fstream>

#include <stdio.h>
#include <legion.h>
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  LEAF_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

class InlineTracingMapper : public DefaultMapper {
public:
  InlineTracingMapper(Machine m,
      Runtime *rt, Processor p)
    : DefaultMapper(rt->get_mapper_runtime(), m, p)
  {
  }
public:
  virtual void configure_context(MapperContext ctx,
      const Task& task, ContextConfigOutput& output) override
  {
    DefaultMapper::configure_context(ctx, task, output);
    // Set the auto tracing turbo lag to zero so that auto
    // tracing is happening inline in the top-level task
    output.auto_tracing_turbo_lag = 0;
  }
public:
  static void register_my_mapper(Machine m, Runtime *rt,
      const std::set<Processor> &local_procs)
  {
    for (Processor proc: local_procs)
      rt->replace_default_mapper(new InlineTracingMapper(m, rt, proc), proc);
  }
};

void leaf_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  // Nothing to do 
}

// Simple LFSR for deterministic random numbers
uint16_t lfsr_step(uint16_t lfsr)
{
  // Example: taps at bits 16 and 14 (polynomial x^16 + x^14 + 1)
  uint16_t lsb = lfsr & 1;
  lfsr >>= 1;
  if (lsb)
    lfsr ^= 0xB400;
  return lfsr;
}

void top_level_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  // Make two regions
  IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 3));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(uint64_t), FID_DATA);
  }
  LogicalRegion one = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion two = runtime->create_logical_region(ctx, is, fs);

  // Fill them
  const uint64_t value = 42;
  runtime->fill_field(ctx, one, one, FID_DATA, &value, sizeof(value));
  runtime->fill_field(ctx, two, two, FID_DATA, &value, sizeof(value));

  // Run a thousand iterations of the leaf task
  TaskLauncher launcher(LEAF_TASK_ID, UntypedBuffer());
  RegionRequirement& req = launcher.add_region_requirement(
      RegionRequirement(one, LEGION_READ_WRITE, LEGION_EXCLUSIVE, one));
  req.flags = LEGION_SUPPRESS_WARNINGS_FLAG;
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(two, LEGION_READ_ONLY, LEGION_EXCLUSIVE, two));
  launcher.add_field(1, FID_DATA);

  uint16_t lfsr = 0xACE1; // Any non-zero seed
  DiscardLauncher discard(one, one);
  discard.add_field(FID_DATA);

  for (unsigned idx = 0; idx < 1000; idx++)
  {
    runtime->execute_task(ctx, launcher);
    // Periodically insert a random discard operation to keep the trace honest
    lfsr = lfsr_step(lfsr);
    if ((lfsr % 10) == 0)
      runtime->discard_fields(ctx, discard);
  }
  
  // Clean up
  runtime->destroy_logical_region(ctx, one);
  runtime->destroy_logical_region(ctx, two);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv)
{
  pid_t child = fork();
  if (child == 0) {
    // Run legion in the child process
    {
      TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_replicable();
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
      TaskVariantRegistrar registrar(LEAF_TASK_ID, "leaf_task");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<leaf_task>(registrar, "leaf_task");
    }
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Runtime::add_registration_callback(InlineTracingMapper::register_my_mapper);
    return Runtime::start(argc, argv);
  } else {
    // The parent process checks the output of the logfile
    // Find the name of the logfile to check
    std::optional<int> file_index;
    for (int idx = 0; idx < argc; idx++)
    {
      if (strcmp(argv[idx],"-logfile") != 0)
        continue;
      file_index = ++idx;
      break;
    }
    // Wait for this child process to finish
    int status = 0;
    pid_t finished = waitpid(child, &status, 0);
    assert(finished == child);
    if (status != 0)
    {
      printf("Child process crashed\n");
      return 1;
    }
    // Must have a logfile
    if (!file_index)
    {
      printf("Missing logfile in command line arguments\n");
      return 1;
    }
    std::string filename(argv[*file_index]);
    // Check if there is a '%' in the file name, if so replace it
    // with '0'. We only need to check on this on node zero anyway
    // as tracing will be the same across all nodes with control replication.
    for (unsigned idx = 0; idx < filename.size(); idx++)
    {
      if (filename[idx] == '%')
        filename[idx] = '0';
    }
    std::ifstream file(filename);
    if (!file.is_open())
    {
      printf("Could not open file %s\n", argv[*file_index]);
      return 1;
    }
    std::string line;
    bool found_tracing = false;
    while (std::getline(file, line))
    {
      if (line.find("Replaying trace") == std::string::npos)
        continue;
      found_tracing = true;
      break;
    }
    file.close();
    if (!found_tracing)
    {
      printf("Did not find any evidence of auto tracing\n");
      return 1;
    }
    return 0;
  }
}
