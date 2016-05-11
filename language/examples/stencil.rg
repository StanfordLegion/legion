-- Copyright 2016 Stanford University
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

-- Inspired by https://github.com/ParRes/Kernels/tree/master/LEGION/Stencil

import "regent"

local c = regentlib.c

local min = regentlib.fmin
local max = regentlib.fmax

fspace point {
  input : double,
  output : double,
}

terra make_bloated_rect(rect : c.legion_rect_2d_t, radius : int)
  return c.legion_rect_2d_t {
    lo = c.legion_point_2d_t {
      x = arrayof(c.coord_t, rect.lo.x[0] - radius, rect.lo.x[1] - radius) },
    hi = c.legion_point_2d_t {
      x = arrayof(c.coord_t, rect.hi.x[0] - radius, rect.hi.x[1] - radius) },
  }
end

terra to_domain_point(x : int2d) : c.legion_domain_point_t
  return [int2d:to_domain_point(`x)]
end

terra to_rect(lo : int2d, hi : int2d) : c.legion_rect_2d_t
  return c.legion_rect_2d_t {
    lo = [int2d:to_point(`lo)],
    hi = [int2d:to_point(`hi)],
  }
end

task make_tile_partition(points : region(ispace(int2d), point),
                         tiles : ispace(int2d),
                         n : int64, nt : int64)
  var coloring = c.legion_domain_point_coloring_create()
  for i in tiles do
    var lo = int2d { x = i.x * n / nt, y = i.y * n / nt }
    var hi = int2d { x = (i.x + 1) * n / nt - 1, y = (i.y + 1) * n / nt - 1 }
    var rect = to_rect(lo, hi)
    c.legion_domain_point_coloring_color_domain(
      coloring, to_domain_point(i), c.legion_domain_from_rect_2d(rect))
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_interior_partition(points : region(ispace(int2d), point),
                         tiles : ispace(int2d),
                         n : int64, nt : int64, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  for i in tiles do
    var lo = int2d { x = max(radius, i.x * n / nt), y = max(radius, i.y * n / nt) }
    var hi = int2d { x = min(n - radius, (i.x + 1) * n / nt) - 1, y = min(n - radius, (i.y + 1) * n / nt) - 1 }
    var rect = to_rect(lo, hi)
    c.legion_domain_point_coloring_color_domain(
      coloring, to_domain_point(i), c.legion_domain_from_rect_2d(rect))
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_bloated_partition(points : region(ispace(int2d), point),
                            tiles : ispace(int2d),
                            private : partition(disjoint, points, tiles),
                            radius : int)
  var coloring = c.legion_domain_point_coloring_create()
  for i in tiles do
    var pts = private[i]
    var rect = c.legion_domain_get_rect_2d(
      c.legion_index_space_get_domain(
        __runtime(), __context(), (__raw(pts)).index_space))
    var bloated = make_bloated_rect(rect, radius)
    c.legion_domain_point_coloring_color_domain(
      coloring, to_domain_point(i), c.legion_domain_from_rect_2d(bloated))
  end
  var p = partition(aliased, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

__demand(__inline)
task off(i : int2d, x : int, y : int)
  return int2d { x = i.x - x, y = i.y - y }
end

task stencil(private : region(ispace(int2d), point), ghost : region(ispace(int2d), point))
where reads writes(private.output), reads(ghost.input) do
  for i in private do
    private[i].output = ghost[i].input +
      -0.5*ghost[off(i, -1, -1)].input +
      -0.5*ghost[off(i, -1,  1)].input +
       0.5*ghost[off(i,  1, -1)].input +
       0.5*ghost[off(i,  1,  1)].input
  end
end

task increment(points : region(ispace(int2d), point))
where reads writes(points.input) do
  for i in points do
    points[i].input = points[i].input + 1
  end
end

task check(points : region(ispace(int2d), point), ts : int64)
where reads(points.{input, output}) do
  for i in points do
    regentlib.assert(points[i].input == ts, "test failed")
    regentlib.assert(points[i].output == ts - 1, "test failed")
  end
end

task main()
  var n : int64, nt : int64, ts : int64 = 10, 4, 10
  var radius : int64 = 1
  var grid = ispace(int2d, { x = n, y = n })
  var tiles = ispace(int2d, { x = nt, y = nt })

  var points = region(grid, point)
  var private = make_tile_partition(points, tiles, n, nt)
  var interior = make_interior_partition(points, tiles, n, nt, radius)
  var ghost = make_bloated_partition(points, tiles, interior, radius)

  fill(points.{input, output}, 0)

  for t = 0, ts do
    for i in tiles do
      stencil(interior[i], ghost[i])
    end
    for i in tiles do
      increment(private[i])
    end
  end

  for i in tiles do
    check(interior[i], ts)
  end
end
regentlib.start(main)
