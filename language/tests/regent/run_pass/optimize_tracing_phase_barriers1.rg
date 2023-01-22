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
--  [ "-dm:memoize", "-ll:cpu", "2" ],
--  [ "-dm:memoize", "-ll:cpu", "2", "-lg:no_fence_elision" ],
--  [ "-dm:memoize", "-ll:cpu", "2", "-lg:no_trace_optimization" ]
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

  local mapper_cc = root_dir .. "optimize_tracing_phase_barriers.cc"
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    mapper_so = out_dir .. "liboptimize_tracing_phase_barriers.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "liboptimize_tracing_phase_barriers.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "stencil_mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  regentlib.linklibrary(mapper_so)
  cmapper = terralib.includec("optimize_tracing_phase_barriers.h", include_dirs)
end

task f1(r : region(ispace(int1d), int),
        a : phase_barrier,
        b : phase_barrier)
where reads writes(r), awaits(a), arrives(b) do
  for e in r do
    regentlib.assert(@e == 0, "test failed in f1")
    @e = 1
  end
end

task f2(r : region(ispace(int1d), int),
        a : phase_barrier,
        b : phase_barrier)
where reads writes(r), awaits(a), arrives(b) do
  for e in r do
    regentlib.assert(@e == 0, "test failed in f2")
    @e = 1
  end
end

task g1(r : region(ispace(int1d), int),
        a : phase_barrier,
        b : phase_barrier)
where reads writes(r), arrives(a), awaits(b) do
  for e in r do
    regentlib.assert(@e == 1, "test failed in g1")
    @e = 0
  end
end

task g2(r : region(ispace(int1d), int),
        a : phase_barrier,
        b : phase_barrier)
where reads writes(r), arrives(a), awaits(b) do
  for e in r do
    regentlib.assert(@e == 1, "test failed in g2")
    @e = 0
  end
end

__forbid(__inner)
task t1(r : region(ispace(int1d), int),
        p : partition(disjoint, r, ispace(int1d)),
        a : phase_barrier,
        b : phase_barrier)
where reads writes simultaneous(r)
do
  __demand(__trace)
  for k = 0, 5 do
    __demand(__index_launch)
    for c in p.colors do
      f1(p[c], a, b)
    end
    a = advance(a)
    b = advance(b)

    __demand(__index_launch)
    for c in p.colors do
      f2(p[c], a, b)
    end
    a = advance(a)
    b = advance(b)
  end
end

__forbid(__inner)
task t2(r : region(ispace(int1d), int),
        p : partition(disjoint, r, ispace(int1d)),
        a : phase_barrier,
        b : phase_barrier)
where reads writes simultaneous(r)
do
  __demand(__trace)
  for k = 0, 5 do
    b = advance(b)
    __demand(__index_launch)
    for c in p.colors do
      g1(p[c], a, b)
    end
    a = advance(a)

    b = advance(b)
    __demand(__index_launch)
    for c in p.colors do
      g2(p[c], a, b)
    end
    a = advance(a)
  end
end

task main()
  var r = region(ispace(int1d, 5000000), int)
  var p = partition(equal, r, ispace(int1d, 2))
  var a = phase_barrier(1)
  var b = phase_barrier(1)
  fill(r, 0)
  must_epoch
    t1(r, p, a, b)
    t2(r, p, a, b)
  end
end

regentlib.start(main, cmapper.register_mappers)
