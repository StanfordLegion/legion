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

assert(regentlib.config["separate"], "test requires separate compilation")

local format = require("std/format")

struct fs {
  x : int
  y : int
  z : int
}

extern task my_regent_task(r : region(ispace(int1d), fs), x : int, y : double, z : bool)
where reads writes(r.{x, y}), reads(r.z) end


task other_regent_task(r : region(ispace(int1d), fs), s : region(ispace(int1d), fs))
where reads writes(r.{x, y}, s.z), reads(r.z, s.x), reduces+(s.y) do
  format.println("Task with two region requirements")
  my_regent_task(r, 3, 4, false)
end

-- Save tasks to libseparate_compilation_tasks_part2.so
local root_dir = arg[0]:match(".*/") or "./"
local separate_compilation_tasks_part2_h = root_dir .. "separate_compilation_tasks_part2.h"
local separate_compilation_tasks_part2_so = root_dir .. "libseparate_compilation_tasks_part2.so"
regentlib.save_tasks(separate_compilation_tasks_part2_h, separate_compilation_tasks_part2_so)
