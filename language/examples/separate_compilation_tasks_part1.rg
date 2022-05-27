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

local SAME_ADDRESS_SPACE = 4 -- (1 << 2)

struct fs {
  x : int
  y : int
  z : int
}

task my_regent_task(r : region(ispace(int1d), fs), x : int, y : double, z : bool)
where reads writes(r.{x, y}), reads(r.z) do
  format.println("Hello from Regent! (values {} {e} {})", x, y, z)
end
my_regent_task:set_mapper_id(0) -- default mapper
my_regent_task:set_mapping_tag_id(SAME_ADDRESS_SPACE)

-- Save tasks to libseparate_compilation_tasks_part1.so
local root_dir = arg[0]:match(".*/") or "./"
local separate_compilation_tasks_part1_h = root_dir .. "separate_compilation_tasks_part1.h"
local separate_compilation_tasks_part1_so = root_dir .. "libseparate_compilation_tasks_part1.so"
-- Test with launcher interface disabled, since technically it shouldn't be required.
regentlib.save_tasks(separate_compilation_tasks_part1_h, separate_compilation_tasks_part1_so, nil, nil, nil, nil, false)
