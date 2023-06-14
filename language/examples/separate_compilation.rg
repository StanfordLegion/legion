-- Copyright 2023 Stanford University
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

-- Make sure this all happens in a temporary directory in case we're
-- running concurrently.
local tmp_dir
do
  -- use os.tmpname to get a hopefully-unique directory to work in
  local tmpfile = os.tmpname()
  tmp_dir = tmpfile .. ".d/"
  assert(os.execute("mkdir " .. tmp_dir) == 0)
  os.remove(tmpfile)  -- remove this now that we have our directory
end

-- Compile separate tasks.
local root_dir = arg[0]:match(".*/") or "./"
local loaders = terralib.newlist()
local link_libraries = terralib.newlist({"-L" .. tmp_dir})
for _, part in ipairs({"tasks_part1", "tasks_part2", "main"}) do
  local regent_exe = os.getenv('REGENT') or 'regent'
  local tasks_rg = "separate_compilation_" .. part .. ".rg"
  assert(os.execute("cp " .. root_dir .. tasks_rg .. " " .. tmp_dir .. tasks_rg) == 0)
  local tasks_h = "separate_compilation_" .. part .. ".h"
  local tasks_lib = "-lseparate_compilation_" .. part
  if os.execute(regent_exe .. " " .. tmp_dir .. tasks_rg .. " -fseparate 1") ~= 0 then
    print("Error: failed to compile " .. tmp_dir .. tasks_rg)
    assert(false)
  end
  local tasks_c = terralib.includec(tasks_h, {"-I", tmp_dir})
  loaders:insert(tasks_c["separate_compilation_" .. part .. "_h_register"])
  link_libraries:insert(tasks_lib)
end

-- Link code copied from regentlib.save_tasks.
local use_cmake = os.getenv("USE_CMAKE") == "1"
local lib_dir = os.getenv("LG_RT_DIR") .. "/../bindings/regent"
if use_cmake then
  lib_dir = os.getenv("CMAKE_BUILD_DIR") .. "/lib"
end
link_libraries:insertall({"-L" .. lib_dir, "-lregent"})
if use_cmake then
  link_libraries:insertall({"-llegion", "-lrealm"})
end

terra main(argc : int, argv : &rawstring)
  escape
    for i, thunk in ipairs(loaders) do
      if i ~= #loaders then
        emit quote thunk() end
      else
        emit quote thunk(argc, argv) end
      end
    end
  end
end

local executable = root_dir .. "separate_compilation.exe"
terralib.saveobj(executable, {main=main}, link_libraries)

local args = rawget(_G, "arg")
local executable_args = terralib.newlist()
for _, arg in ipairs(args) do
  executable_args:insert(arg)
end

local ffi = require("ffi")
local cmd
if ffi.os == "OSX" then
  local lib_path = (os.getenv("DYLD_LIBRARY_PATH") or "") .. ":"
  cmd = "DYLD_LIBRARY_PATH=" .. lib_path .. ":" .. lib_dir .. ":" .. tmp_dir .. " " .. executable .. " " .. executable_args:concat(" ")
else
  local lib_path = (os.getenv("LD_LIBRARY_PATH") or "") .. ":"
  cmd = "LD_LIBRARY_PATH=" .. lib_path .. ":" .. lib_dir .. ":" .. tmp_dir .. " " .. executable .. " " .. executable_args:concat(" ")
end
print(cmd)
assert(os.execute(cmd) == 0)

-- os.execute("rm -r " .. tmp_dir)
