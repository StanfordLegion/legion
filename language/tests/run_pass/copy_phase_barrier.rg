-- Copyright 2015 Stanford University
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

-- runs-with:
-- []

-- [["-ll:cpu", "2"]]

import "regent"

-- Compile and link copy_phase_barrier.cc
local cmapper
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = root_dir .. "../../../runtime/"
  local legion_dir = runtime_dir .. "legion/"
  local mapper_dir = runtime_dir .. "mappers/"
  local mapper_cc = root_dir .. "copy_phase_barrier.cc"
  local mapper_so = os.tmpname() .. ".so" -- root_dir .. "copy_phase_barrier.so"
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -std=c++0x -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  terralib.linklibrary(mapper_so)
  cmapper = terralib.includec(
    "copy_phase_barrier.h",
    {"-I", root_dir, "-I", runtime_dir, "-I", mapper_dir, "-I", legion_dir})
end

task mul(r : region(int), y : int, t : phase_barrier)
where reads writes(r), awaits(t) do
  for x in r do
    @x *= y
  end
end

task f(r : region(int), s : region(int), t : phase_barrier)
where reads writes simultaneous(r, s), no_access_flag(s) do
  copy(r, s, arrives(t))
end

task g(r : region(int), s : region(int), t : phase_barrier)
where reads writes simultaneous(r, s), no_access_flag(r) do
  var t1 = advance(t)
  mul(s, 2, t1)
  copy(s, r, +, awaits(t1)) -- redundant, but need to test
end

task k() : int
  var r = region(ispace(ptr, 5), int)
  var x = new(ptr(int, r))
  var s = region(ispace(ptr, 5), int)
  var y = new(ptr(int, s))

  @x = 123
  @y = 456

  var t = phase_barrier(1)
  must_epoch
    f(r, s, t)
    g(r, s, t)
  end

  return @x
end

task main()
  regentlib.assert(k() == 369, "test failed")
end
cmapper.register_mappers()
regentlib.start(main)
