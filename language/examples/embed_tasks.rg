-- Copyright 2022 Stanford University
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

-- This file is not meant to be run directly.

-- runs-with:
-- []

import "regent"

local root_dir = arg[0]:match(".*/") or "./"
local c = terralib.includec("embed.h", {"-I", root_dir})

local fs = c.fs -- Get the field space from C header

task unexposed_task(r : region(ispace(int1d), fs), p : partition(disjoint, r, ispace(int1d)))
  regentlib.c.printf("Unexposed task!\n")
end


task my_regent_task(r : region(ispace(int1d), fs), x : int, y : double, z : bool, w : float[4])
where reads writes(r.{x, y}), reads(r.z) do
  regentlib.c.printf("Hello from Regent! (values %d %e %d [%.1f, %.1f, %.1f, %.1f])\n", x, y, z, w[0], w[1], w[2], w[3])
  var p = partition(equal, r, ispace(int1d, 2))
  unexposed_task(r, p)
end


task other_regent_task(r : region(ispace(int1d), fs), s : region(ispace(int1d), fs))
where reads writes(r.{x, y}, s.z), reads(r.z, s.x), reduces+(s.y) do
  regentlib.c.printf("Task with two region requirements\n")
end

__demand(__inner)
task inline_regent_task(r : region(ispace(int1d), fs))
where reads writes(r.{x, y, z}) do
  regentlib.c.printf("Inline inner task\n")
  var w : float[4]
  w[0] = 1.1
  w[1] = 2.1
  w[2] = 3.1
  w[3] = 4.1
  my_regent_task(r, 0, 0.0, false, w)
end

local embed_tasks_dir
if os.getenv('SAVEOBJ') == '1' then
  embed_tasks_dir = root_dir
else
  -- use os.tmpname to get a hopefully-unique directory to work in
  local tmpfile = os.tmpname()
  embed_tasks_dir = tmpfile .. ".d/"
  local res = os.execute("mkdir " .. embed_tasks_dir)
  assert(res == 0)
  os.remove(tmpfile)  -- remove this now that we have our directory
end

local task_whitelist = {}
task_whitelist["my_regent_task"] = my_regent_task
task_whitelist["other_regent_task"] = other_regent_task
task_whitelist["inline_regent_task"] = inline_regent_task

local embed_tasks_h = embed_tasks_dir .. "embed_tasks.h"
local embed_tasks_so = embed_tasks_dir .. "libembed_tasks.so"
regentlib.save_tasks(embed_tasks_h, embed_tasks_so, nil, nil, "embed_tasks_register", task_whitelist)
return embed_tasks_dir
