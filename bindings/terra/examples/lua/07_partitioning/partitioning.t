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
  local num_subregions = 4

  -- See if we have any command line arguments to parse
  -- Note we now have a new command line parameter which specifies
  -- how many subregions we should make.
  local command_args = legion:get_input_args()
  for i = 1, #command_args do
    if command_args[i] == '-n' then
      num_elements = tonumber(command_args[i+1])
    end
    if command_args[i] == '-b' then
      num_subregions  = tonumber(command_args[i+1])
    end
  end
  printf("Running daxpy for %d elements...", num_elements)
  printf("Partitioning data into %d sub-regions...", num_subregions)

  -- Create our logical regions using the same schemas as earlier examples
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

  -- In addition to using rectangles and domains for launching index spaces
  -- of tasks (see example 02), Legion also uses them for performing
  -- operations on logical regions.  Here we create a rectangle and a
  -- corresponding domain for describing the space of subregions that we
  -- want to create.  Each subregion is assigned a 'color' which is why
  -- we name the variables 'color_bounds' and 'color_domain'.  We'll use
  -- these below when we partition the region.
  local color_bounds = Rect:new(Point:new{0}, Point:new{num_subregions-1})
  local color_domain = Domain:from_rect(color_bounds)

  -- Parallelism in Legion is implicit.  This means that rather than
  -- explicitly saying what should run in parallel, Legion applications
  -- partition up data and tasks specify which regions they access.
  -- The Legion runtime computes non-interference as a function of
  -- regions, fields, and privileges and then determines which tasks
  -- are safe to run in parallel.
  --
  -- Data partitioning is performed on index spaces.  The paritioning
  -- operation is used to break an index space of points into subsets
  -- of points each of which will become a sub index space.  Partitions
  -- created on an index space are then transitively applied to all the
  -- logical regions created using the index space.  We will show how
  -- to get names to the subregions later in this example.
  --
  -- Here we want to create the IndexPartition 'ip'.  We'll illustrate
  -- two ways of creating an index partition depending on whether the
  -- array being partitioned can be evenly partitioned into subsets
  -- or not.  There are other methods to partitioning index spaces
  -- which are not covered here.  We'll cover the case of coloring
  -- individual points in an index space in our capstone circuit example.
  local ip = nil
  if (num_elements % num_subregions) ~= 0 then
    -- Not evenly divisible
    -- Create domain coloring to store the coloring
    -- maps from colors to domains for each subregion
    --
    -- In this block of code we handle the case where the index space
    -- of points is not evenly divisible by the desired number of
    -- subregions.  This gives us the opporunity to illustrate a
    -- general approach to coloring arrays.  The general idea is
    -- to create a map from colors to sub-domains.  Colors correspond
    -- to the sub index spaces that will be made and each sub-domain
    -- describes the set points to be kept in that domain.

    -- Computer upper and lower bounds on the number of elements per subregion
    local lower_bound = math.floor(num_elements/num_subregions)
    local upper_bound = lower_bound+1
    local number_small = num_subregions - (num_elements % num_subregions)
    -- Create a coloring object to store the domain coloring.  A
    -- DomainColoring type is internally a typedef of an STL map from Colors
    -- (unsigned integers) to Domain objects and can be found in
    -- legion_types.h along with type declarations for all user-visible
    -- Legion types.
    local coloring = DomainColoring:new()
    local index = 0

    -- We fill in the coloring by computing the domain of points
    -- to assign to each color.  We assign 'elmts_per_subregion'
    -- to all colors except the last one when we clamp the
    -- value to the maximum number of elements.
    for color = 0, num_subregions - 1 do
      local num_elmts =
        ite(color < number_small, lower_bound, upper_bound)
      assert((index+num_elmts) <= num_elements)
      local subrect = Rect:new(Point:new{index}, Point:new{index+num_elmts-1})
      coloring[color] = Domain:from_rect(subrect)
      index = index + num_elmts
    end
    -- Once we have computed our domain coloring we are now ready
    -- to create the partition.  Creating a partitiong simply
    -- invovles giving the Legion runtime an index space to
    -- partition 'is', a domain of colors, and then a domain
    -- coloring.  In addition the application must specify whether
    -- the given partition is disjoint or not.  This example is
    -- a disjoint partition because there are no overlapping
    -- points between any of the sub index spaces.  In debug mode
    -- the runtime will check whether disjointness assertions
    -- actually hold.  In the next example we'll see an
    -- application which makes use of a non-disjoint partition.
    ip = runtime:create_index_partition(ctx, is, color_domain,
                                        coloring, true)
  else
    -- In the case where we know that the number of subregions
    -- evenly divides the number of elements, the Array namespace
    -- in Legion provide productivity constructs for partitioning
    -- an array.  Blockify is one example of these constructs.
    -- A Blockify will evenly divide a rectangle into subsets
    -- containing the specified number of elements in each
    -- dimension.  Since we are only dealing with a 1-D rectangle,
    -- we need only specify the number of elements to have
    -- in each subset.  A Blockify object is mapping from colors
    -- to subsets of an index space the same as a DomainColoring,
    -- but is implicitly disjoint.  The 'create_index_partition'
    -- method on the Legion runtime is overloaded a different
    -- instance of it takes mappings like Blockify and returns
    -- an IndexPartition.
    local coloring = Blockify:new{num_elements/num_subregions}
    ip = runtime:create_index_partition(ctx, is, coloring)
  end

  -- The index space 'is' was used in creating two logical regions: 'input_lr'
  -- and 'output_lr'.  By creating an IndexPartitiong of 'is' we implicitly
  -- created a LogicalPartition for each of the logical regions created using
  -- 'is'.  The Legion runtime provides several ways of getting the names for
  -- these LogicalPartitions.  We'll look at one of them here.  The
  -- 'get_logical_partition' method takes a LogicalRegion and an IndexPartition
  -- and returns the LogicalPartition of the given LogicalRegion that corresponds
  -- to the given IndexPartition.
  local input_lp = runtime:get_logical_partition(ctx, input_lr, ip)
  local output_lp = runtime:get_logical_partition(ctx, output_lr, ip)

  -- Create our launch domain.  Note that is the same as color domain
  -- as we are going to launch one task for each subregion we created.
  local launch_domain = color_domain
  local arg_map = ArgumentMap:new()

  -- As in previous examples, we now want to launch tasks for initializing
  -- both the fields.  However, to increase the amount of parallelism
  -- exposed to the runtime we will launch separate sub-tasks for each of
  -- the logical subregions created by our partitioning.  To express this
  -- we create an IndexLauncher for launching an index space of tasks
  -- the same as example 02.
  local init_launcher = IndexLauncher:new(INIT_FIELD_TASK_ID, launch_domain,
                                          nil, arg_map)
  -- For index space task launches we don't want to have to explicitly
  -- enumerate separate region requirements for all points in our launch
  -- domain.  Instead Legion allows applications to place an upper bound
  -- on privileges required by subtasks and then specify which privileges
  -- each subtask receives using a projection function.  In the case of
  -- the field initialization task, we say that all the subtasks will be
  -- using some subregion of the LogicalPartition 'input_lp'.  Applications
  -- may also specify upper bounds using logical regions and not partitions.
  --
  -- The Legion implementation assumes that all all points in an index
  -- space task launch request non-interfering privileges and for performance
  -- reasons this is unchecked.  This means if two tasks in the same index
  -- space are accessing aliased data, then they must either both be
  -- with read-only or reduce privileges.
  --
  -- When the runtime enumerates the launch_domain, it will invoke the
  -- projection function for each point in the space and use the resulting
  -- LogicalRegion computed for each point in the index space of tasks.
  -- The projection ID '0' is reserved and corresponds to the identity
  -- function which simply zips the space of tasks with the space of
  -- subregions in the partition.  Applications can register their own
  -- projections functions via the 'register_region_projection' and
  -- 'register_partition_projection' functions before starting
  -- the runtime similar to how tasks are registered.
  init_launcher:add_region_requirement(
    RegionRequirement:new(input_lp, 0, legion.WRITE_DISCARD,
                          legion.EXCLUSIVE, input_lr))
  init_launcher:add_field(0, FID_X)
  runtime:execute_index_space(ctx, init_launcher)

  -- Modify our region requirement to intiialize the other field
  -- in the same way.  Note that after we do this we have exposed
  -- 2*num_subregions task-level parallelism to the runtime because
  -- we have launched tasks that are both data-parallel on
  -- sub-regions and task-parallel on accessing different fields.
  -- The power of Legion is that it allows programmers to express
  -- these data usage patterns and automatically extracts both
  -- kinds of parallelism in a unified programming framework.
  init_launcher.region_requirements[0].privilege_fields:clear()
  init_launcher.region_requirements[0].instance_fields:clear()
  init_launcher:add_field(0, FID_Y)

  runtime:execute_index_space(ctx, init_launcher)

  local alpha = drand48()
  -- We launch the subtasks for performing the daxpy computation
  -- in a similar way to the initialize field tasks.  Note we
  -- again make use of two RegionRequirements which use a
  -- partition as the upper bound for the privileges for the task.
  local arg = TaskArgument:new(alpha, double)
  local daxpy_launcher = IndexLauncher:new(DAXPY_TASK_ID, launch_domain,
                                          arg, arg_map)
  daxpy_launcher:add_region_requirement(
    RegionRequirement:new(input_lp, 0, legion.READ_ONLY,
                          legion.EXCLUSIVE, input_lr))
  daxpy_launcher:add_field(0, FID_X)
  daxpy_launcher:add_field(0, FID_Y)
  daxpy_launcher:add_region_requirement(
    RegionRequirement:new(output_lp, 0, legion.WRITE_DISCARD,
                          legion.EXCLUSIVE, output_lr))
  daxpy_launcher:add_field(1, FID_Z)
  runtime:execute_index_space(ctx, daxpy_launcher)

  -- While we could also issue parallel subtasks for the checking
  -- task, we only issue a single task launch to illustrate an
  -- important Legion concept.  Note the checking task operates
  -- on the entire 'input_lr' and 'output_lr' regions and not
  -- on the subregions.  Even though the previous tasks were
  -- all operating on subregions, Legion will correctly compute
  -- data dependences on all the subtasks that generated the
  -- data in these two regions.
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

  runtime:destroy_logical_partition(ctx, input_lp)
  runtime:destroy_logical_partition(ctx, output_lp)
  runtime:destroy_logical_region(ctx, input_lr)
  runtime:destroy_logical_region(ctx, output_lr)
  runtime:destroy_field_space(ctx, input_fs)
  runtime:destroy_field_space(ctx, output_fs)
  runtime:destroy_index_space(ctx, is)
end

function init_field_task(task, regions, ctx, runtime)
  assert(regions.size == 1)
  assert(task.regions.size == 1)
  assert(task.regions[0].privilege_fields.size == 1)

  local fid = task.regions[0].privilege_fields[0]
  local point = task.index_point.point_data[0]
  printf("Initializing field %d for block %d...", fid, point)

  local acc = regions[0]:get_field_accessor(fid):typeify(double)

  -- Note here that we get the domain for the subregion for
  -- this task from the runtime which makes it safe for running
  -- both as a single task and as part of an index space of tasks.
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
  local point = task.index_point.point_data[0]

  local acc_x = regions[0]:get_field_accessor(FID_X):typeify(double)
  local acc_y = regions[0]:get_field_accessor(FID_Y):typeify(double)
  local acc_z = regions[1]:get_field_accessor(FID_Z):typeify(double)
  printf("Running daxpy computation with alpha %.8g for point %d...",
          alpha, point);

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
  -- Note we mark that all of these tasks are capable of being
  -- run both as single tasks and as index space tasks
  legion:register_lua_task_void("init_field_task", INIT_FIELD_TASK_ID,
    legion.LOC_PROC, true, true)
  legion:register_lua_task_void("daxpy_task", DAXPY_TASK_ID,
    legion.LOC_PROC, true, true)
  legion:register_lua_task_void("check_task", CHECK_TASK_ID,
    legion.LOC_PROC, true, true)
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
