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
local std = terralib.includec("stdio.h")

legion:import_types()

HELLO_WORLD_ID = 1

terra hello_world_task(task : task_t,
                       regions : &physical_region_t,
                       num_regions : uint32,
                       ctx : context_t,
                       runtime : runtime_t)
  std.printf("Hello World!\n")
end

function legion_main(arg)
  legion:set_top_level_task_id(HELLO_WORLD_ID)
  legion:register_terra_task_void(
    hello_world_task,
    HELLO_WORLD_ID, legion.LOC_PROC,
    true,  -- single
    false  -- index
  )
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
