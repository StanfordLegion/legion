-- Copyright 2024 Stanford University
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
--   ["-ll:cpu", "4", "-fflow-spmd", "1"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

-- This tests the SPMD optimization of the compiler with:
--   * keeping multiple partitions up-to-date

fspace st {
  x : int,
  y : int,
}

task a(r : region(st))
where reads writes(r.x), reads(r.y) do
  for e in r do
    e.x = e.y + 1
  end
end

task b(r : region(st), s : region(st), t : region(st), u : region(st))
where reads(r.x, s.x, t.x, u.x), writes(r.y) do
  for e in r do
    r[e].y = 2*r[e].x + 3*s[e].x + 4*t[e].x + 5*u[e].x
  end
end

task main()
  var cs = ispace(int1d, 3)
  var r = region(ispace(ptr, 5), st)
  var e0 = dynamic_cast(ptr(st, r), 0)
  var e1 = dynamic_cast(ptr(st, r), 1)
  var e2 = dynamic_cast(ptr(st, r), 2)
  var e3 = dynamic_cast(ptr(st, r), 3)
  var e4 = dynamic_cast(ptr(st, r), 4)

  -- These have to be separate partitions to trigger the bug.
  var m = partition(equal, r, cs)
  var n = partition(equal, r, cs)
  var o = partition(equal, r, cs)
  var p = partition(equal, r, cs)

  for e in r do
    e.x = 0
    e.y = int(e)
  end

  __demand(__spmd)
  do
    for i = 0, 3 do
      for c in cs do
        a(m[c])
      end
      for c in cs do
        b(m[c], n[c], o[c], p[c])
      end
    end
  end

  for e in r do
    regentlib.c.printf("e %d: %d %d\n", e, e.x, e.y)
  end

  regentlib.assert(e0.x == 211, "test failed")
  regentlib.assert(e1.x == 407, "test failed")
  regentlib.assert(e2.x == 603, "test failed")
  regentlib.assert(e3.x == 799, "test failed")
  regentlib.assert(e4.x == 995, "test failed")

  -- FIXME: This breaks SPMD.
  -- regentlib.assert(e0.y ==  2954, "test failed")
  -- regentlib.assert(e1.y ==  5698, "test failed")
  -- regentlib.assert(e2.y ==  8442, "test failed")
  -- regentlib.assert(e3.y == 11186, "test failed")
  -- regentlib.assert(e4.y == 13930, "test failed")
end
regentlib.start(main)
