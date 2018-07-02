-- Copyright 2018 Stanford University
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

task test_1d()
  var is = ispace(int1d, 2)
  var is_bounds = is.bounds
  var x : int1d, y : int1d = -1, 2
  var a, b, c, d, e, f = x + y, x - y, x * y, x / y, x % y, x % is_bounds

  var t = y <= is_bounds

  var f1, f2, f3 = x <= x + y, y <= is_bounds, y - 1 <= is_bounds

  regentlib.assert([int](a) == 1, "test failed")
  regentlib.assert([int](b) == -3, "test failed")
  regentlib.assert([int](c) == -2, "test failed")
  regentlib.assert([int](d) == 0, "test failed")
  regentlib.assert([int](e) == -1, "test failed")
  regentlib.assert([int](f) == 1, "test failed")
  regentlib.assert(f1 and (not f2) and f3, "test failed")

  var r = region(is, int)
  for e in r do
    regentlib.assert(e <= r.bounds, "test failed")
  end

  var z, w = unsafe_cast(int1d(int, r), x), unsafe_cast(int1d(int, r), y)
  a, b, c, d, e, f = z + w, z - y, x * w, z / w, x % w, z % is_bounds
  regentlib.assert([int](a) == 1, "test failed")
  regentlib.assert([int](b) == -3, "test failed")
  regentlib.assert([int](c) == -2, "test failed")
  regentlib.assert([int](d) == 0, "test failed")
  regentlib.assert([int](e) == -1, "test failed")
  regentlib.assert([int](f) == 1, "test failed")
end

task test_2d()
  var is = ispace(int2d, { 2, 4 })
  var is_bounds = is.bounds
  var x : int2d, y : int2d = { -1, -2 }, { 2, 3 }
  var a, b, c, d, e, f = x + y, x - y, x * y, x / y, x % y, x % is_bounds

  var f1, f2, f3 = x <= x + y, y <= is_bounds, y - {1, 1} <= is_bounds

  regentlib.assert(a.x == 1, "test failed")   regentlib.assert(a.y == 1, "test failed")
  regentlib.assert(b.x == -3, "test failed")  regentlib.assert(b.y == -5, "test failed")
  regentlib.assert(c.x == -2, "test failed")  regentlib.assert(c.y == -6, "test failed")
  regentlib.assert(d.x == 0, "test failed")   regentlib.assert(d.y == 0, "test failed")
  regentlib.assert(e.x == -1, "test failed")  regentlib.assert(e.y == -2, "test failed")
  regentlib.assert(f.x == 1, "test failed")   regentlib.assert(f.y == 2, "test failed")
  regentlib.assert(f1 and (not f2) and f3, "test failed")

  var r = region(is, int)
  for e in r do
    regentlib.assert(e <= r.bounds, "test failed")
  end

  var z, w = unsafe_cast(int2d(int, r), x), unsafe_cast(int2d(int, r), y)
  a, b, c, d, e, f = z + w, z - y, x * w, z / w, x % w, z % is_bounds
  regentlib.assert(a.x == 1, "test failed")   regentlib.assert(a.y == 1, "test failed")
  regentlib.assert(b.x == -3, "test failed")  regentlib.assert(b.y == -5, "test failed")
  regentlib.assert(c.x == -2, "test failed")  regentlib.assert(c.y == -6, "test failed")
  regentlib.assert(d.x == 0, "test failed")   regentlib.assert(d.y == 0, "test failed")
  regentlib.assert(e.x == -1, "test failed")  regentlib.assert(e.y == -2, "test failed")
  regentlib.assert(f.x == 1, "test failed")   regentlib.assert(f.y == 2, "test failed")
end

task test_3d()
  var is = ispace(int3d, { 2, 4, 7 })
  var is_bounds = is.bounds
  var x : int3d, y : int3d = { -1, -2, 12 }, { 2, 3, 4 }
  var a, b, c, d, e, f = x + y, x - y, x * y, x / y, x % y, x % is_bounds

  var f1, f2, f3 = x <= x + y, y <= is_bounds, y - {1, 1, 1} <= is_bounds

  regentlib.assert(a.x == 1, "test failed")   regentlib.assert(a.y == 1, "test failed")  regentlib.assert(a.z == 16, "test failed")
  regentlib.assert(b.x == -3, "test failed")  regentlib.assert(b.y == -5, "test failed") regentlib.assert(b.z == 8, "test failed")
  regentlib.assert(c.x == -2, "test failed")  regentlib.assert(c.y == -6, "test failed") regentlib.assert(c.z == 48, "test failed")
  regentlib.assert(d.x == 0, "test failed")   regentlib.assert(d.y == 0, "test failed")  regentlib.assert(d.z == 3, "test failed")
  regentlib.assert(e.x == -1, "test failed")  regentlib.assert(e.y == -2, "test failed") regentlib.assert(e.z == 0, "test failed")
  regentlib.assert(f.x == 1, "test failed")   regentlib.assert(f.y == 2, "test failed")  regentlib.assert(f.z == 5, "test failed")
  regentlib.assert(f1 and (not f2) and f3, "test failed")

  var r = region(is, int)
  for e in r do
    regentlib.assert(e <= r.bounds, "test failed")
  end

  var z, w = unsafe_cast(int3d(int, r), x), unsafe_cast(int3d(int, r), y)
  a, b, c, d, e, f = z + w, z - y, x * w, z / w, x % w, z % is_bounds
  regentlib.assert(a.x == 1, "test failed")   regentlib.assert(a.y == 1, "test failed")  regentlib.assert(a.z == 16, "test failed")
  regentlib.assert(b.x == -3, "test failed")  regentlib.assert(b.y == -5, "test failed") regentlib.assert(b.z == 8, "test failed")
  regentlib.assert(c.x == -2, "test failed")  regentlib.assert(c.y == -6, "test failed") regentlib.assert(c.z == 48, "test failed")
  regentlib.assert(d.x == 0, "test failed")   regentlib.assert(d.y == 0, "test failed")  regentlib.assert(d.z == 3, "test failed")
  regentlib.assert(e.x == -1, "test failed")  regentlib.assert(e.y == -2, "test failed") regentlib.assert(e.z == 0, "test failed")
  regentlib.assert(f.x == 1, "test failed")   regentlib.assert(f.y == 2, "test failed")  regentlib.assert(f.z == 5, "test failed")
end

task main()
  test_1d()
  test_2d()
  test_3d()
end
regentlib.start(main)
