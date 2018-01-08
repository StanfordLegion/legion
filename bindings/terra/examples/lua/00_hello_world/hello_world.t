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

-- We declare the IDs for user-level tasks in the global scope
HELLO_WORLD_ID = 1

-- All Legion task written in Lua must have this signature.
function hello_world_task(task, regions, ctx, runtime)
  -- A task runs just like a normal Lua function.
  print "Hello World!"
end

-- As there is no pre-defined entry function for Lua, we write a separate main
-- function called legion_main, which should be called only once from the first
-- interpreter executed from the command line.  Once we start the runtime, it
-- will begin running the top-level task.
function legion_main(arg)
  -- Before starting the Legion runtime, you first have to tell it
  -- what the ID is for the top-level task.
  legion:set_top_level_task_id(HELLO_WORLD_ID)
  -- Before starting the Legion runtime, all possible tasks that the
  -- runtime can potentially run must be registered with the runtime.
  -- The function pointer is passed as a template argument.  The second
  -- argument specifies the kind of processor on which the task can be
  -- run: latency optimized cores (LOC) aka CPUs or throughput optimized
  -- cores (TOC) aka GPUs.  The last two arguments specify whether the
  -- task can be run as a single task or an index space task (covered
  -- in more detail in later examples).  The top-level task must always
  -- be able to be run as a single task.
  legion:register_lua_task_void(
    "hello_world_task",
    HELLO_WORLD_ID, legion.LOC_PROC,
    true,  -- single
    false -- index
  )
  -- Now we're ready to start the runtime, so tell it to begin the
  -- execution.
  legion:start(arg)
end

-- The following lines of code are an idiom to detect whether the current
-- interpreter is executed from the command line, only in which case
-- the arg global variable is set to a list of command line arguments.
if rawget(_G, "arg") then
  legion_main(arg)
end
