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

-- In this example we illustrate how the Legion
-- programming model supports multiple partitions
-- of the same logical region and the benefits it
-- provides by allowing multiple views onto the
-- same logical region.  We compute a simple 5-point
-- 1D stencil using the standard forumala:
-- f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/12h
-- For simplicity we'll assume h=1.

TOP_LEVEL_TASK_ID = 0
INIT_FIELD_TASK_ID = 1
STENCIL_TASK_ID = 2
CHECK_TASK_ID = 3

FID_VAL = 0
FID_DERIV = 1

function top_level_task(task, regions, ctx, runtime)
  local num_elements = 1024
  local num_subregions = 4

  local command_args = legion:get_input_args()
  -- Check for any command line arguments
  for i = 1, #command_args do
    if command_args[i] == '-n' then
      num_elements = tonumber(command_args[i+1])
    end
    if command_args[i] == '-b' then
      num_subregions  = tonumber(command_args[i+1])
    end
  end
  printf("Running stencil computation for %d elements...", num_elements)
  printf("Partitioning data into %d sub-regions...", num_subregions)

  -- For this example we'll create a single logical region with two
  -- fields.  We'll initialize the field identified by 'FID_VAL' with
  -- our input data and then compute the derivatives stencil values
  -- and write them into the field identified by 'FID_DERIV'.
  local elem_rect = Rect:new(Point:new{0},Point:new{num_elements-1})
  local is = runtime:create_index_space(ctx, Domain:from_rect(elem_rect))
  local fs = runtime:create_field_space(ctx)
  do
    local allocator =
      runtime:create_field_allocator(ctx, fs)
    allocator:allocate_field(sizeof(double),FID_VAL)
    allocator:allocate_field(sizeof(double),FID_DERIV)
  end
  local stencil_lr = runtime:create_logical_region(ctx, is, fs)

  -- Make our color_domain based on the number of subregions
  -- that we want to create.
  local color_bounds = Rect:new(Point:new{0}, Point:new{num_subregions-1})
  local color_domain = Domain:from_rect(color_bounds)

  -- In this example we need to create two partitions: one disjoint
  -- partition for describing the output values that are going to
  -- be computed by each sub-task that we launch and a second
  -- aliased partition which will describe the input values needed
  -- for performing each task.  Note that for the second partition
  -- each subregion will be a superset of its corresponding region
  -- in the first partition, but will also require two 'ghost' cells
  -- on each side.  The need for these ghost cells means that the
  -- subregions in the second partition will be aliased.
  local disjoint_ip, ghost_ip
  do
    local lower_bound = math.floor(num_elements/num_subregions)
    local upper_bound = lower_bound+1
    local number_small = num_subregions - (num_elements % num_subregions)
    local disjoint_coloring = DomainColoring:new()
    local ghost_coloring = DomainColoring:new()
    local index = 0
    -- Iterate over all the colors and compute the entry
    -- for both partitions for each color.
    for color = 0, num_subregions - 1 do
      local num_elmts =
        ite(color < number_small, lower_bound, upper_bound)
      assert((index+num_elmts) <= num_elements)
      local subrect = Rect:new(Point:new{index}, Point:new{index+num_elmts-1})
      disjoint_coloring[color] = Domain:from_rect(subrect)
      -- Now compute the points assigned to this color for
      -- the second partition.  Here we need a superset of the
      -- points that we just computed including the two additional
      -- points on each side.  We handle the edge cases by clamping
      -- values to their minimum and maximum values.  This creates
      -- four cases of clamping both above and below, clamping below,
      -- clamping above, and no clamping.
      if index < 2 then
        if index+num_elmts+2 > num_elements then
          local ghost_rect = Rect:new(Point:new{0},Point:new{num_elements-1})
          ghost_coloring[color] = Domain:from_rect(ghost_rect)
        else
          local ghost_rect = Rect:new(Point:new{0},Point:new{index+num_elmts+1})
          ghost_coloring[color] = Domain:from_rect(ghost_rect)
        end
      else
        if index+num_elmts+2 > num_elements then
          local ghost_rect = Rect:new(Point:new{index-2},Point:new{num_elements-1})
          ghost_coloring[color] = Domain:from_rect(ghost_rect)
        else
          local ghost_rect = Rect:new(Point:new{index-2},Point:new{index+num_elmts+1})
          ghost_coloring[color] = Domain:from_rect(ghost_rect)
        end
      end
      index = index + num_elmts
    end
    -- Once we've computed both of our colorings then we can
    -- create our partitions.  Note that we tell the runtime
    -- that one is disjoint will the second one is not.
    disjoint_ip = runtime:create_index_partition(ctx, is, color_domain,
                                                 disjoint_coloring, true)
    ghost_ip = runtime:create_index_partition(ctx, is, color_domain,
                                              ghost_coloring, true)
  end

  -- Once we've created our index partitions, we can get the
  -- corresponding logical partitions for the stencil_lr
  -- logical region.
  local disjoint_lp = runtime:get_logical_partition(ctx, stencil_lr, disjoint_ip)
  local ghost_lp = runtime:get_logical_partition(ctx, stencil_lr, ghost_ip)

  -- Our launch domain will again be isomorphic to our coloring domain.
  local launch_domain = color_domain
  local arg_map = ArgumentMap:new()

  -- First initialize the 'FID_VAL' field with some data
  local init_launcher = IndexLauncher:new(INIT_FIELD_TASK_ID, launch_domain,
                                          nil, arg_map)
  init_launcher:add_region_requirement(
    RegionRequirement:new(disjoint_lp, 0, legion.WRITE_DISCARD,
                          legion.EXCLUSIVE, stencil_lr))
  init_launcher:add_field(0, FID_VAL)
  runtime:execute_index_space(ctx, init_launcher)

  -- Now we're going to launch our stencil computation.  We
  -- specify two region requirements for the stencil task.
  -- Each region requirement is upper bounded by one of our
  -- two partitions.  The first region requirement requests
  -- read-only privileges on the ghost partition.  Note that
  -- because we are only requesting read-only privileges, all
  -- of our sub-tasks in the index space launch will be
  -- non-interfering.  The second region requirement asks for
  -- read-write privileges on the disjoint partition for
  -- the 'FID_DERIV' field.  Again this meets with the
  -- mandate that all points in our index space task
  -- launch be non-interfering.
  local arg = TaskArgument:new(num_elements, int)
  local stencil_launcher = IndexLauncher:new(STENCIL_TASK_ID, launch_domain,
                                             arg, arg_map)
  stencil_launcher:add_region_requirement(
    RegionRequirement:new(ghost_lp, 0, legion.READ_ONLY,
                          legion.EXCLUSIVE, stencil_lr))
  stencil_launcher:add_field(0, FID_VAL)
  stencil_launcher:add_region_requirement(
    RegionRequirement:new(disjoint_lp, 0, legion.READ_WRITE,
                          legion.EXCLUSIVE, stencil_lr))
  stencil_launcher:add_field(1, FID_DERIV)
  runtime:execute_index_space(ctx, stencil_launcher)

  -- Finally, we launch a single task to check the results.
  local check_launcher = TaskLauncher:new(CHECK_TASK_ID, arg)
  check_launcher:add_region_requirement(
    RegionRequirement:new(stencil_lr, legion.READ_ONLY,
                          legion.EXCLUSIVE, stencil_lr))
  check_launcher:add_field(0, FID_VAL)
  check_launcher:add_region_requirement(
    RegionRequirement:new(stencil_lr, legion.READ_WRITE,
                          legion.EXCLUSIVE, stencil_lr))
  check_launcher:add_field(1, FID_DERIV)
  runtime:execute_task(ctx, check_launcher)

  -- Clean up our region, index space, and field space
  runtime:destroy_logical_partition(ctx, disjoint_lp)
  runtime:destroy_logical_partition(ctx, ghost_lp)
  runtime:destroy_logical_region(ctx, stencil_lr)
  runtime:destroy_field_space(ctx, fs)
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

  local dom = runtime:get_index_space_domain(ctx,
    task.regions[0].region:get_index_space())
  local rect = dom:get_rect()
  local pir = GenericPointInRectIterator:new(rect);
  while pir:has_next() do
    acc:write(DomainPoint:from_point(pir.p), drand48())
    pir:next()
  end
end

-- Our stencil tasks is interesting because it
-- has both slow and fast versions depending
-- on whether or not its bounds have been clamped.
function stencil_task(task, regions, ctx, runtime)
  assert(regions.size == 2)
  assert(task.regions.size == 2)
  assert(task.regions[0].privilege_fields.size == 1)
  assert(task.regions[1].privilege_fields.size == 1)

  local max_elements = task:get_args(int)
  local point = task.index_point.point_data[0]

  local read_fid = task.regions[0].privilege_fields[0]
  local write_fid = task.regions[1].privilege_fields[0]

  local read_acc = regions[0]:get_field_accessor(read_fid):typeify(double)
  local write_acc = regions[1]:get_field_accessor(write_fid):typeify(double)

  local dom = runtime:get_index_space_domain(ctx,
    task.regions[1].region:get_index_space())
  local rect = dom:get_rect()

  local zero = DomainPoint:from_point(Point:new{0})
  local max = DomainPoint:from_point(Point:new{max_elements-1})
  local one = Point:new{1}
  local two = Point:new{2}
  -- If we are on the edges of the entire space we are 
  -- operating over, then we're going to do the slow
  -- path which checks for clamping when necessary.
  -- If not, then we can do the fast path without
  -- any checks.
  if (rect.lo[0] < 2) or (rect.hi[0] > (max_elements-3)) then
    printf("Running slow stencil path for point %d...", point)
    -- Note in the slow path that there are checks which
    -- perform clamps when necessary before reading values.
    local pir = GenericPointInRectIterator:new(rect);
    while pir:has_next() do
      local l2, l1, r1, r2
      if pir.p[0] < 2 then
        l2 = read_acc:read(zero)
      else
        l2 = read_acc:read(DomainPoint:from_point(pir.p-two))
      end
      if pir.p[0] < 1 then
        l1 = read_acc:read(zero)
      else
        l1 = read_acc:read(DomainPoint:from_point(pir.p-one))
      end
      if pir.p[0] > (max_elements-2) then
        r1 = read_acc:read(max)
      else
        r1 = read_acc:read(DomainPoint:from_point(pir.p+one))
      end
      if pir.p[0] > (max_elements-3) then
        r2 = read_acc:read(max)
      else
        r2 = read_acc:read(DomainPoint:from_point(pir.p+two))
      end

      local result = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0
      write_acc:write(DomainPoint:from_point(pir.p), result)
      pir:next()
    end
  else
    printf("Running fast stencil path for point %d...", point)
    -- In the fast path, we don't need any checks
    local pir = GenericPointInRectIterator:new(rect);
    while pir:has_next() do
      local l2 = read_acc:read(DomainPoint:from_point(pir.p-two))
      local l1 = read_acc:read(DomainPoint:from_point(pir.p-one))
      local r1 = read_acc:read(DomainPoint:from_point(pir.p+one))
      local r2 = read_acc:read(DomainPoint:from_point(pir.p+two))

      local result = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0
      write_acc:write(DomainPoint:from_point(pir.p), result)
      pir:next()
    end
  end
end

function check_task(task, regions, ctx, runtime)
  assert(regions.size == 2)
  assert(task.regions.size == 2)
  assert(task.regions[0].privilege_fields.size == 1)
  assert(task.regions[1].privilege_fields.size == 1)
  local max_elements = task:get_args(int)

  local src_fid = task.regions[0].privilege_fields[0]
  local dst_fid = task.regions[1].privilege_fields[0]

  local src_acc = regions[0]:get_field_accessor(src_fid):typeify(double)
  local dst_acc = regions[1]:get_field_accessor(dst_fid):typeify(double)

  local dom = runtime:get_index_space_domain(ctx,
    task.regions[1].region:get_index_space())
  local rect = dom:get_rect()
  local zero = DomainPoint:from_point(Point:new{0})
  local max = DomainPoint:from_point(Point:new{max_elements-1})
  local one = Point:new{1}
  local two = Point:new{2}

  -- This is the checking task so we can just do the slow path
  local all_passed = true
  local pir = GenericPointInRectIterator:new(rect);
  while pir:has_next() do
    local l2, l1, r1, r2
    if pir.p[0] < 2 then
      l2 = src_acc:read(zero)
    else
      l2 = src_acc:read(DomainPoint:from_point(pir.p-two))
    end
    if pir.p[0] < 1 then
      l1 = src_acc:read(zero)
    else
      l1 = src_acc:read(DomainPoint:from_point(pir.p-one))
    end
    if pir.p[0] > (max_elements-2) then
      r1 = src_acc:read(max)
    else
      r1 = src_acc:read(DomainPoint:from_point(pir.p+one))
    end
    if pir.p[0] > (max_elements-3) then
      r2 = src_acc:read(max)
    else
      r2 = src_acc:read(DomainPoint:from_point(pir.p+two))
    end

    local expected = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0
    local received = dst_acc:read(DomainPoint:from_point(pir.p))
    -- Probably shouldn't bitwise compare floating point
    -- numbers but the order of operations are the same so they
    -- should be bitwise equal.
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
    legion.LOC_PROC, true, true)
  legion:register_lua_task_void("stencil_task", STENCIL_TASK_ID,
    legion.LOC_PROC, true, true)
  legion:register_lua_task_void("check_task", CHECK_TASK_ID,
    legion.LOC_PROC, true, true)
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
