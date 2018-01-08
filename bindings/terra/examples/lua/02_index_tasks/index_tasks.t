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

TOP_LEVEL_TASK_ID = 0
HELLO_WORLD_INDEX_ID = 1

-- This example is a redux version of hello world
-- which shows how launch a large array of tasks
-- using a single runtime call.  We also describe
-- the basic Legion types for arrays, domains,
-- and points and give examples of how they work.

function top_level_task(task, reigons, ctx, runtime)
  local num_points = 4
  -- See how many points to run
  local command_args = legion:get_input_args()
  if #command_args >= 1 then
    local i = 1
    while i <= #command_args do
      local arg = command_args[i]
      if string.sub(arg, 1, 1) == "-" then
        i = i + 2
      else
        num_points = tonumber(arg)
        break
      end
    end
    assert(num_points > 0)
  end
  print("Running hello world redux for " .. num_points .. " points...")

  -- To aid in describing structured data, Legion supports
  -- a Rect type which is used to describe an array of
  -- points.  To specify a Rect a user gives two Points
  -- which specify the lower and upper bounds on each dimension
  -- respectively.  Here we create a 1-D Rect which
  -- we'll use to launch an array of tasks.  Note that the
  -- bounds on Rects are inclusive.
  local launch_bounds = Rect:new(Point:new{0}, Point:new{num_points-1})
  -- Rects can be converted to Domains.   Users can easily convert
  -- between Domains and Rects using the 'from_rect' and
  -- 'get_rect' methods.  Most Legion runtime calls will
  -- take Domains.
  local launch_domain = Domain:from_rect(launch_bounds)

  -- When we go to launch a large group of tasks in a single
  -- call, we may want to pass different arguments to each
  -- task.  ArgumentMaps allow the user to pass different
  -- arguments to different points.  Note that ArgumentMaps
  -- do not need to specify arguments for all points.  Legion
  -- is intelligent about only passing arguments to the tasks
  -- that have them.  Here we pass some values that we'll
  -- use to illustrate how values get returned from an index
  -- task launch.
  local arg_map = ArgumentMap:new()
  for i = 0, num_points - 1 do
    local input = i + 10
    local arg = TaskArgument:new(input, int)
    arg_map:set_point(DomainPoint:from_point(Point:new{i}), arg)
  end

  -- Legion supports launching an array of tasks with a
  -- single call.  We call these index tasks as we are launching
  -- an array of tasks with one task for each point in the
  -- array.  Index tasks are launched similar to single
  -- tasks by using an index task launcher.  IndexLauncher
  -- objects take the additional arguments of an ArgumentMap,
  -- a TaskArgument which is a global argument that will
  -- be passed to all tasks launched, and a domain describing
  -- the points to be launched.
  local index_launcher =
    IndexLauncher:new(HELLO_WORLD_INDEX_ID,
                      launch_domain,
                      nil,
                      arg_map)
  -- Index tasks are launched the same as single tasks, but
  -- return a future map which will store a future for all
  -- points in the index space task launch.  Application
  -- tasks can either wait on the future map for all tasks
  -- in the index space to finish, or it can pull out
  -- individual futures for specific points on which to wait.
  local fm = runtime:execute_index_space(ctx, index_launcher)
  -- Here we wait for all the futures to be ready
  fm:wait_all_results()
  -- Now we can check that the future results that came back
  -- from all the points in the index task are double
  -- their input.
  local all_passed = true
  for i = 0, num_points - 1 do
    local expected = 2*(i+10)
    local received = fm:get_result(int, DomainPoint:from_point(Point:new{i}))
    if expected ~= received then
      print(
        string.format("Check failed for point %d: %d != %d",
        i, expected, received))
      all_passed = false
    end
  end
  if all_passed then
    print("All checks passed")
  end
end

function index_space_task(task, regions, ctx, runtime)
  -- The point for this task is available in the task
  -- structure under the 'index_point' field.
  assert(task.index_point.dim == 1)
  printf("Hello world from task %d!", task.index_point.point_data[0])
  -- Values passed through an argument map are available
  -- through the 'get_local_args' method which accepts the value type
  -- as the first argument.
  local input = task:get_local_args(int)
  return (2*input)
end

function legion_main(arg)
  legion:set_top_level_task_id(TOP_LEVEL_TASK_ID)
  legion:register_lua_task_void("top_level_task",
    TOP_LEVEL_TASK_ID, legion.LOC_PROC, true,  false)
  legion:register_lua_task(int, "index_space_task",
    HELLO_WORLD_INDEX_ID, legion.LOC_PROC, false,  true)
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
