-- Copyright 2017 Stanford University
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

import "regent"

local root_dir = arg[0]:match(".*/") or "./"
local c = terralib.includec("embed.h", {"-I", root_dir})

local fs = c.fs -- Get the field space from C

task my_regent_task(r : region(fs))
where reads writes(r) do
  regentlib.c.printf("Hello from Regent!\n")
end

-- Save tasks to libembed_tasks.so
local embed_tasks_lib = "embed_tasks"
do
  local embed_tasks_h = root_dir .. "embed_tasks.h"
  local embed_tasks_so = root_dir .. "libembed_tasks.so"
  regentlib.save_tasks(embed_tasks_h, embed_tasks_so)
end

-- Compile and execute embed.cc
local exe
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv("LG_RT_DIR") .. "/"
  local legion_dir = runtime_dir .. "legion/"
  local mapper_dir = runtime_dir .. "mappers/"
  local realm_dir = runtime_dir .. "realm/"
  local binding_dir = root_dir .. "../../bindings/terra/"

  local embed_cc = root_dir .. "embed.cc"
  if os.getenv('SAVEOBJ') == '1' then
    exe = root_dir .. "embed.exe"
  else
    exe = os.tmpname()
  end

  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -Wall -Werror"

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " ..
                " -I " .. realm_dir .. " " ..  embed_cc ..
                 " -L " .. root_dir .. " " .. " -l" .. embed_tasks_lib .. " " ..
                 " -L " .. binding_dir .. " -llegion_terra " ..
                 " -o " .. exe)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. embed_cc)
    assert(false)
  end
end

local env = ""
if os.getenv("DYLD_LIBRARY_PATH") then
  env = "DYLD_LIBRARY_PATH=" .. os.getenv("DYLD_LIBRARY_PATH") .. ":" .. root_dir .. " "
elseif os.getenv("LD_LIBRARY_PATH") then
  env = "LD_LIBRARY_PATH=" .. os.getenv("LD_LIBRARY_PATH") .. ":" .. root_dir .. " "
end

-- Pass the arguments along so that the child process is able to
-- complete the execution of the parent.
local args = ""
for _, arg in ipairs(rawget(_G, "arg")) do
  args = args .. " " .. arg
end

assert(os.execute(env .. exe .. args) == 0)
