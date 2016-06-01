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
      x = arrayof(c.coord_t, rect.hi.x[0] + radius, rect.hi.x[1] + radius) },
  }
end

terra to_rect(lo : int2d, hi : int2d) : c.legion_rect_2d_t
  return c.legion_rect_2d_t {
    lo = lo:to_point(),
    hi = hi:to_point(),
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
      coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
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
      coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
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
      coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(bloated))
  end
  var p = partition(aliased, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

local function off(i, x, y)
  return rexpr int2d { x = i.x + x, y = i.y + y } end
end

local function make_stencil_pattern(points, index, off_x, off_y, radius)
  local value
  for i = 1, radius do
    local neg = off_x < 0 or off_y < 0
    local coeff = ((neg and -1) or 1)/(2*i*radius)
    local x, y = off_x*i, off_y*i
    local component = rexpr coeff*points[ [off(index, x, y)] ].input end
    if value then
      value = rexpr value + component end
    else
      value = rexpr component end
    end
  end
  return value
end

local function make_stencil(radius)
  local task st(private : region(ispace(int2d), point), ghost : region(ispace(int2d), point))
  where reads writes(private.output), reads(ghost.input) do
    for i in private do
      private[i].output = ghost[i].input +
        [make_stencil_pattern(ghost, i,  0, -1, radius)] +
        [make_stencil_pattern(ghost, i, -1,  0, radius)] +
        [make_stencil_pattern(ghost, i,  1,  0, radius)] +
        [make_stencil_pattern(ghost, i,  0,  1, radius)]
    end
  end
  return st
end

local RADIUS = 2
local stencil = make_stencil(RADIUS)

task increment(points : region(ispace(int2d), point))
where reads writes(points.input) do
  for i in points do
    points[i].input = points[i].input + 1
  end
end

task check(points : region(ispace(int2d), point), tsteps : int64, init : int64)
where reads(points.{input, output}) do
  var expect_in = init + tsteps
  var expect_out = init + tsteps - 1
  for i in points do
    if points[i].input ~= expect_in then
      for i2 in points do
        c.printf("input (%lld,%lld): %.0f should be %lld\n",
                 i2.x, i2.y, points[i2].input, expect_in)
      end
    end
    regentlib.assert(points[i].input == expect_in, "test failed")
    if points[i].output ~= expect_out then
      for i2 in points do
        c.printf("output (%lld,%lld): %.0f should be %lld\n",
                 i2.x, i2.y, points[i2].output, expect_out)
      end
    end
    regentlib.assert(points[i].output == expect_out, "test failed")
  end
end

task main()
  var n : int64 = 12
  var nt : int64 = 4
  var tsteps : int64 = 10
  var init : int64 = 1000

  var radius : int64 = RADIUS
  var grid = ispace(int2d, { x = n, y = n })
  var tiles = ispace(int2d, { x = nt, y = nt })

  var points = region(grid, point)
  var private = make_tile_partition(points, tiles, n, nt)
  var interior = make_interior_partition(points, tiles, n, nt, radius)
  var ghost = make_bloated_partition(points, tiles, interior, radius)

  fill(points.{input, output}, init)

  for t = 0, tsteps do
    for i in tiles do
      stencil(interior[i], ghost[i])
    end
    for i in tiles do
      increment(private[i])
    end
  end

  for i in tiles do
    check(interior[i], tsteps, init)
  end
end
regentlib.start(main)
