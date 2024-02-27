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

local launcher = require("std/launcher")
local cmapper = launcher.build_library("optimize_tracing_phase_barriers")

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
        a : phase_barrier,
        b : phase_barrier)
where reads writes simultaneous(r)
do
  __demand(__trace)
  for k = 0, 5 do
    f1(r, a, b)
    a = advance(a)
    b = advance(b)

    f2(r, a, b)
    a = advance(a)
    b = advance(b)
  end
end

__forbid(__inner)
task t2(r : region(ispace(int1d), int),
        a : phase_barrier,
        b : phase_barrier)
where reads writes simultaneous(r)
do
  __demand(__trace)
  for k = 0, 5 do
    b = advance(b)
    g2(r, a, b)
    a = advance(a)

    b = advance(b)
    g1(r, a, b)
    a = advance(a)
  end
end

task main()
  var r = region(ispace(int1d, 5000000), int)
  var a = phase_barrier(1)
  var b = phase_barrier(1)
  fill(r, 0)
  must_epoch
    t1(r, a, b)
    t2(r, a, b)
  end
end

launcher.launch(main, "optimize_tracing_phase_barriers2", cmapper.register_mappers, {"-loptimize_tracing_phase_barriers"})
