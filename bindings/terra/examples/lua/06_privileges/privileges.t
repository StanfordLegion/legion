-- Copyright 2018 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

require("legionlib")
local std = terralib.includec("stdlib.h")
rawset(_G, "drand48", std.drand48)

TOP_LEVEL_TASK_ID = 0
INIT_FIELD_TASK_ID = 1
DAXPY_TASK_ID = 2
CHECK_TASK_ID = 3

FID_X = 0
FID_Y = 1
FID_Z = 2

function top_level_task(task, regions, ctx, runtime)
  local num_elements = 1024
  -- See if we have any command line arguments to parse
  local command_args = legion:get_input_args()
  for i = 1, #command_args do
    if command_args[i] == '-n' then
      num_elements = tonumber(command_args[i+1])
    end
  end
  printf("Running daxpy for %d elements...", num_elements)

  -- Create our logical regions using the same schema that
  -- we used in the previous example.
  local elem_rect = Rect:new(Point:new{0},Point:new{num_elements-1})
  local is = runtime:create_index_space(ctx, Domain:from_rect(elem_rect))
  local input_fs = runtime:create_field_space(ctx)
  do
    local allocator =
      runtime:create_field_allocator(ctx, input_fs)
    allocator:allocate_field(sizeof(double),FID_X)
    allocator:allocate_field(sizeof(double),FID_Y)
  end
  local output_fs = runtime:create_field_space(ctx)
  do
    local allocator =
      runtime:create_field_allocator(ctx, output_fs)
    allocator:allocate_field(sizeof(double),FID_Z)
  end
  local input_lr = runtime:create_logical_region(ctx, is, input_fs)
  local output_lr = runtime:create_logical_region(ctx, is, output_fs)

  -- Instead of using an inline mapping to initialize the fields for
  -- daxpy, in this case we will launch two separate tasks for initializing
  -- each of the fields in parallel.  To launch the sub-tasks for performing
  -- the initialization we again use the launcher objects that were
  -- introduced earlier.  The only difference now is that instead of passing
  -- arguments by value, we now want to specify the logical regions
  -- that the tasks may access as their arguments.  We again make use of
  -- the RegionRequirement struct to name the logical regions and fields
  -- for which the task should have privileges.  In this case we launch
  -- a task that asks for WRITE_DISCARD privileges on the 'X' field.
  --
  -- An important property of the Legion programming model is that sub-tasks
  -- are only allowed to request privileges which are a subset of a
  -- parent task's privileges.  When a task creates a logical region it
  -- is granted full read-write privileges for that logical region.  It
  -- can then pass them down to sub-tasks.  In this example the top-level
  -- task has full privileges on all the fields of input_lr and output_lr.
  -- In this call it passing read-write privileges down to the sub-task
  -- on input_lr on field 'X'.  Legion will enforce the property that the
  -- sub-task only accesses the 'X' field of input_lr.  This property of
  -- Legion is crucial for the implementation of Legion's hierarchical
  -- scheduling algorithm which is described in detail in our two papers.
  local init_launcher = TaskLauncher:new(INIT_FIELD_TASK_ID)
  init_launcher:add_region_requirement(
    RegionRequirement:new(input_lr, legion.WRITE_DISCARD,
                          legion.EXCLUSIVE, input_lr))
  init_launcher:add_field(0, FID_X)
  -- Note that when we launch this task we don't record the future.
  -- This is because we're going to let Legion be responsible for
  -- computing the data dependences between how different tasks access
  -- logical regions.
  runtime:execute_task(ctx, init_launcher)

  -- Re-use the same launcher but with a slightly different RegionRequirement
  -- that requests privileges on field 'Y' instead of 'X'.  Since these
  -- two instances of the init_field_task are accessing different fields
  -- of the input_lr region, they can be run in parallel (whether or not
  -- they do is dependent on the mapping discussed in a later example).
  -- Legion automatically will discover this parallelism since the runtime
  -- understands the fields present on the logical region.
  --
  -- We now call attention to a unique property of the init_field_task.
  -- In this example we've actually called the task with two different
  -- region requirements containing different fields.  The init_field_task
  -- is an example of a field-polymorphic task which is capable of
  -- performing the same operation on different fields of a logical region.
  -- In practice this is very useful property for a task to maintain as
  -- it allows one implementation of a task to be written which is capable
  -- of being used in many places.
  init_launcher.region_requirements[0].privilege_fields:clear()
  init_launcher.region_requirements[0].instance_fields:clear()
  init_launcher:add_field(0, FID_Y)

  runtime:execute_task(ctx, init_launcher)

  -- Now we launch the task to perform the daxpy computation.  We pass
  -- in the alpha value as an argument.  All the rest of the arguments
  -- are RegionRequirements specifying that we are reading the two
  -- fields on the input_lr region and writing results to the output_lr
  -- region.  Legion will automatically compute data dependences
  -- from the two init_field_tasks and will ensure that the program
  -- order execution is obeyed.
  local alpha = drand48()
  local arg = TaskArgument:new(alpha, double)
  local daxpy_launcher = TaskLauncher:new(DAXPY_TASK_ID, arg)
  daxpy_launcher:add_region_requirement(
    RegionRequirement:new(input_lr, legion.READ_ONLY,
                          legion.EXCLUSIVE, input_lr))
  daxpy_launcher:add_field(0, FID_X)
  daxpy_launcher:add_field(0, FID_Y)
  daxpy_launcher:add_region_requirement(
    RegionRequirement:new(output_lr, legion.WRITE_DISCARD,
                          legion.EXCLUSIVE, output_lr))
  daxpy_launcher:add_field(1, FID_Z)

  runtime:execute_task(ctx, daxpy_launcher)

  -- Finally we launch a task to perform the check on the output.  Note
  -- that Legion will compute a data dependence on the first RegionRequirement
  -- with the two init_field_tasks, but not on daxpy task since they
  -- both request read-only privileges on the 'X' and 'Y' fields.  However,
  -- Legion will compute a data dependence on the second region requirement
  -- as the daxpy task was writing the 'Z' field on output_lr and this task
  -- is reading the 'Z' field of the output_lr region.
  local check_launcher = TaskLauncher:new(CHECK_TASK_ID, arg)
  check_launcher:add_region_requirement(
    RegionRequirement:new(input_lr, legion.READ_ONLY,
                          legion.EXCLUSIVE, input_lr))
  check_launcher:add_field(0, FID_X)
  check_launcher:add_field(0, FID_Y)
  check_launcher:add_region_requirement(
    RegionRequirement:new(output_lr, legion.READ_ONLY,
                          legion.EXCLUSIVE, output_lr))
  check_launcher:add_field(1, FID_Z)

  runtime:execute_task(ctx, check_launcher)

  -- Notice that we never once blocked waiting on the result of any sub-task
  -- in the execution of the top-level task.  We don't even block before
  -- destroying any of our resources.  This works because Legion understands
  -- the data being accessed by all of these operations and defers all of
  -- their executions until they are safe to perform.  Legion is still smart
  -- enough to know that the top-level task is not finished until all of
  -- the sub operations that have been performed are completed.  However,
  -- from the programmer's perspective, all of these operations can be
  -- done without ever blocking and thereby exposing as much task-level
  -- parallelism to the Legion runtime as possible.  We'll discuss the
  -- implications of Legion's deferred execution model in a later example.
  runtime:destroy_logical_region(ctx, input_lr)
  runtime:destroy_logical_region(ctx, output_lr)
  runtime:destroy_field_space(ctx, input_fs)
  runtime:destroy_field_space(ctx, output_fs)
  runtime:destroy_index_space(ctx, is)
end

-- Note that tasks get a physical region for every region requriement
-- that they requested when they were launched in the array of 'regions'.
-- In some cases the mapper may have chosen not to map the logical region
-- which means that the task has the necessary privileges to access the
-- region but not a physical instance to access.
function init_field_task(task, regions, ctx, runtime)
  -- Check that the inputs look right since we have no
  -- static checking to help us out.
  assert(regions.size == 1)
  assert(task.regions.size == 1)
  assert(task.regions[0].privilege_fields.size == 1)

  -- This is a field polymorphic function so figure out
  -- which field we are responsible for initializing.
  local fid = task.regions[0].privilege_fields[0]
  local acc = regions[0]:get_field_accessor(fid):typeify(double)
  -- Note that Legion's default mapper always map regions
  -- and the Legion runtime is smart enough not to start
  -- the task until all the regions contain valid data.
  -- Therefore in this case we don't need to call 'wait_until_valid'
  -- on our physical regions and we know that getting this
  -- accessor will never block the task's execution.  If
  -- however we chose to unmap this physical region and then
  -- remap it then we would need to call 'wait_until_valid'
  -- again to ensure that we were accessing valid data.
  local dom = runtime:get_index_space_domain(ctx,
    task.regions[0].region:get_index_space())
  local rect = dom:get_rect()
  local pir = GenericPointInRectIterator:new(rect);
  while pir:has_next() do
    acc:write(DomainPoint:from_point(pir.p), drand48())
    pir:next()
  end
end

function daxpy_task(task, regions, ctx, runtime)
  assert(regions.size == 2)
  assert(task.regions.size == 2)

  local alpha = task:get_args(double)
  local acc_x = regions[0]:get_field_accessor(FID_X):typeify(double)
  local acc_y = regions[0]:get_field_accessor(FID_Y):typeify(double)
  local acc_z = regions[1]:get_field_accessor(FID_Z):typeify(double)
  printf("Running daxpy computation with alpha %.8g...", alpha);
  local dom = runtime:get_index_space_domain(ctx,
    task.regions[0].region:get_index_space())
  local rect = dom:get_rect()

  local pir = GenericPointInRectIterator:new(rect);
  while pir:has_next() do
    local value = alpha * acc_x:read(DomainPoint:from_point(pir.p)) +
                          acc_y:read(DomainPoint:from_point(pir.p))
    acc_z:write(DomainPoint:from_point(pir.p), value)
    pir:next()
  end
end

function check_task(task, regions, ctx, runtime)
  assert(regions.size == 2)
  assert(task.regions.size == 2)

  local alpha = task:get_args(double)
  local acc_x = regions[0]:get_field_accessor(FID_X):typeify(double)
  local acc_y = regions[0]:get_field_accessor(FID_Y):typeify(double)
  local acc_z = regions[1]:get_field_accessor(FID_Z):typeify(double)

  print("Checking results...")
  local dom = runtime:get_index_space_domain(ctx,
    task.regions[0].region:get_index_space())
  local rect = dom:get_rect()
  local all_passed = true
  local pir = GenericPointInRectIterator:new(rect);
  while pir:has_next() do
    local expected  = alpha * acc_x:read(DomainPoint:from_point(pir.p)) +
                              acc_y:read(DomainPoint:from_point(pir.p))
    local received = acc_z:read(DomainPoint:from_point(pir.p))
    -- Probably shouldn't check for floating point equivalence but
    -- the order of operations are the same should they should
    -- be bitwise equal.
    if expected ~= received then
      all_passed = false
    end
    pir:next()
  end

  if all_passed then
    print("SUCCESS!")
  else
    print("FAILURE!")
  end
end

function legion_main(arg)
  legion:set_top_level_task_id(TOP_LEVEL_TASK_ID)
  legion:register_lua_task_void("top_level_task", TOP_LEVEL_TASK_ID,
    legion.LOC_PROC, true, false)
  legion:register_lua_task_void("init_field_task", INIT_FIELD_TASK_ID,
    legion.LOC_PROC, true, false)
  legion:register_lua_task_void("daxpy_task", DAXPY_TASK_ID,
    legion.LOC_PROC, true, false)
  legion:register_lua_task_void("check_task", CHECK_TASK_ID,
    legion.LOC_PROC, true, false)
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
