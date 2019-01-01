-- Copyright 2019 Stanford University
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

task my_regent_task(r : region(ispace(int1d), fs), x : int)
where reads writes(r.{x, y}), reads(r.z) do
  regentlib.c.printf("Hello from Regent! (value %d)\n", x)
end

-- Save tasks to libembed_tasks.so
local embed_tasks_h = root_dir .. "embed_tasks.h"
local embed_tasks_so = root_dir .. "libembed_tasks.so"
regentlib.save_tasks(embed_tasks_h, embed_tasks_so)
return "embed_tasks"
