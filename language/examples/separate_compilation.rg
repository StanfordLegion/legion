-- Copyright 2026 Stanford University
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

-- FIXME (Elliott): debugging https://github.com/StanfordLegion/legion/issues/1514
io.stdout:setvbuf("no")
io.stderr:setvbuf("no")

-- Make sure this all happens in a temporary directory in case we're
-- running concurrently.
local tmp_dir
do
  -- use os.tmpname to get a hopefully-unique directory to work in
  local tmpfile = os.tmpname()
  tmp_dir = tmpfile .. ".d/"
  print("Creating temporary directory " .. tmp_dir)
  assert(os.execute("mkdir " .. tmp_dir) == 0)
  -- Hack: keep the tmpfile to be absolutely sure we won't collide
  -- os.remove(tmpfile)
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
local realm_dir = os.getenv("Realm_ROOT")
if realm_dir then
  link_libraries:insertall({"-L" .. realm_dir .. "/lib"})
end
local lib_dir = os.getenv("LEGION_INSTALL_PREFIX") .. "/lib"
local lib64_dir = os.getenv("LEGION_INSTALL_PREFIX") .. "/lib64"
link_libraries:insertall({"-L" .. lib_dir, "-L" .. lib64_dir, "-lregent", "-llegion", "-lrealm", "-lz"})

local ffi = require("ffi")
if ffi.os == "Linux" then
  link_libraries:insert("-latomic")
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

local main_obj = tmp_dir .. "separate_compilation.o"
local executable = tmp_dir .. "separate_compilation.exe"
terralib.saveobj(main_obj, {main=main})

local cxx = os.getenv("CXX") or "c++"
if os.execute(cxx .. " " .. main_obj .. " " .. link_libraries:concat(" ") .. " -o " .. executable) ~= 0 then
  print("Error: failed to compile " .. executable)
  assert(false)
end

local args = rawget(_G, "arg")
local executable_args = terralib.newlist()
for _, arg in ipairs(args) do
  executable_args:insert(arg)
end
-- FIXME (Elliott): debugging https://github.com/StanfordLegion/legion/issues/1514
executable_args:insert("-lg:registration")
executable_args:insert("-level")
executable_args:insert("runtime=3")

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
