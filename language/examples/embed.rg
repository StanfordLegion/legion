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

import "regent"

local embed_tasks_dir = require("embed_tasks")

-- Compile and execute embed.cc
local exe
local root_dir = arg[0]:match(".*/") or "./"
do
  local binding_dir = root_dir .. "../../bindings/regent/"

  local include_path = ""
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
  end

  local embed_cc = root_dir .. "embed.cc"
  exe = embed_tasks_dir .. "embed.exe"

  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -Wall -Werror " .. (os.getenv('CXXFLAGS') or "")

  local use_cmake = os.getenv("USE_CMAKE") == "1"
  local lib_dir = binding_dir
  local libs = "-lregent"
  if use_cmake then
    lib_dir = os.getenv("CMAKE_BUILD_DIR") .. "/lib"
    libs = libs .. " -llegion -lrealm"
  end
  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 embed_cc ..
		 " -I " .. embed_tasks_dir ..
                 " -L " .. embed_tasks_dir .. " " .. " -lembed_tasks " ..
                 " -L " .. lib_dir .. " " .. libs .. " " ..
                 " -o " .. exe)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. embed_cc)
    assert(false)
  end
end

local env = ""
if os.getenv("DYLD_LIBRARY_PATH") then
  env = "DYLD_LIBRARY_PATH=" .. os.getenv("DYLD_LIBRARY_PATH") .. ":" .. embed_tasks_dir .. " "
elseif os.getenv("LD_LIBRARY_PATH") then
  env = "LD_LIBRARY_PATH=" .. os.getenv("LD_LIBRARY_PATH") .. ":" .. embed_tasks_dir .. " "
end

-- Pass the arguments along so that the child process is able to
-- complete the execution of the parent.
local args = ""
for _, arg in ipairs(rawget(_G, "arg")) do
  args = args .. " " .. arg
end

assert(os.execute(env .. exe .. args) == 0)

-- clean up our mess
if os.getenv('SAVEOBJ') ~= '1' then
  os.execute("rm " .. embed_tasks_dir .. "embed_tasks.h")
  os.execute("rm " .. embed_tasks_dir .. "libembed_tasks.*")
  os.execute("rm " .. embed_tasks_dir .. "embed.exe")
  os.execute("rmdir " .. embed_tasks_dir)
end
