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

-- runs-with:
-- [
--  [ "-dm:memoize" ],
--  [ "-dm:memoize", "-lg:no_fence_elision" ],
--  [ "-dm:memoize", "-lg:no_trace_optimization" ]
-- ]

import "regent"

do
  local root_dir = arg[0]:match(".*/") or "./"

  local include_path = ""
  local include_dirs = terralib.newlist()
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local mapper_cc = root_dir .. "optimize_tracing_invalidate1.cc"
  if os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "liboptimize_tracing_invalidate1.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "optimize_tracing_invalidate1.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  local ffi = require("ffi")
  if ffi.os == "OSX" then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print(cmd)
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  regentlib.linklibrary(mapper_so)
  cmapper = terralib.includec("optimize_tracing_invalidate1.h", include_dirs)
end

fspace fs
{
  input : int;
  output : int;
}

task init(r : region(ispace(int1d), fs))
where reads writes(r)
do
  for e in r do
    e.input = 0
    e.output = 0
  end
end

task inc(r : region(ispace(int1d), fs))
where reads(r.input), writes(r.output)
do
  for e in r do
    e.output = e.input + 1
  end
end

task step(r : region(ispace(int1d), fs))
where writes(r.input), reads(r.output)
do
  for e in r do
    e.input = e.output
  end
end

task check(r : region(ispace(int1d), fs), n : int)
where reads(r)
do
  for e in r do
    regentlib.assert(e.input % 3 == n, "test, failed")
  end
end

task main()
  var n = 2
  var r = region(ispace(int1d, n), fs)
  var cs = ispace(int1d, n)
  var p = partition(equal, r, cs)
  var q = partition(equal, r, cs)

  for color in cs do init(p[color]) end
  for k = 0, 3 do
    __demand(__trace)
    for i = 0, 2 do
      for color in cs do inc(p[color]) end
      for color in cs do step(p[color]) end
    end
    for color in cs do check(q[color], 2 * (k + 1) % 3) end
  end
  for color in cs do init(p[color]) end
  for k = 0, 3 do
    __demand(__trace)
    for i = 0, 2 do
      for color in cs do inc(p[color]) end
      for color in cs do step(p[color]) end
      for color in cs do inc(q[color]) end
      for color in cs do step(q[color]) end
    end
    for color in cs do check(q[color], 4 * (k + 1) % 3) end
  end
  for color in cs do init(p[color]) end
  for k = 0, 2 do
    __demand(__trace)
    do
      for color in cs do inc(p[color]) end
      for color in cs do step(p[color]) end
    end
    __demand(__trace)
    do
      for color in cs do inc(q[color]) end
      for color in cs do step(q[color]) end
    end
  end
  for color in cs do check(p[color], 1) end
end
regentlib.start(main, cmapper.register_mappers)
