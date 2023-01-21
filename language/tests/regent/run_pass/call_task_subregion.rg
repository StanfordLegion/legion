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

local c = regentlib.c

task f(s : region(int), y : ptr(int, s)) : int
where reads(s) do
  return @y
end

task h(s : region(int), y : ptr(int, s)) : int
where reads(s), writes(s) do
  @y += 100
  return @y
end

task k(s : region(int)) : int
where reads(s), writes(s) do
  for y in s do
    @y += 10000
  end
  return 0
end

task g() : int
  var r = region(ispace(ptr, 1), int)
  var rc = c.legion_coloring_create()
  var x = dynamic_cast(ptr(int, r), 0)
  c.legion_coloring_add_point(rc, 0, __raw(x))
  c.legion_coloring_add_point(rc, 1, __raw(x))
  var p = partition(aliased, r, rc)
  var r0 = p[0]
  var r1 = p[1]
  var p0 = partition(aliased, r0, rc)
  var r00 = p0[0]
  var r01 = p0[1]
  c.legion_coloring_destroy(rc)
  var x0 = dynamic_cast(ptr(int, r0), x)
  var x00 = dynamic_cast(ptr(int, r00), x)
  var x01 = dynamic_cast(ptr(int, r01), x)
  var x1 = dynamic_cast(ptr(int, r1), x)
  @x = 5

  -- Open.
  f(r0, x0)
  for i = 0, 1 do
    k(p[i])
  end

  -- Close.
  f(r, x)
  h(r, x)

  -- Close and reopen (write).
  h(r0, x0)
  h(r1, x1)

  -- Close and reopen (read).
  f(r, x)
  f(r0, x0)
  f(r00, x00)
  h(r1, x1)

  for i = 0, 5 do
    h(r00, x00)
    h(r01, x01)
  end
  return f(r0, x0)
end

task main()
  regentlib.assert(g() == 11405, "test failed")
end
regentlib.start(main)
