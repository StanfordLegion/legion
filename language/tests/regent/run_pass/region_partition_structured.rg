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

import "regent"

local c = regentlib.c

task inc(is : ispace(int1d), r : region(is, int), y : int)
where reads writes(r) do
  for i in is do
    r[i] += y
  end
end

task sum(is : ispace(int1d), r : region(is, int))
where reads(r) do
  var t = 0
  for i in is do
    t += r[i]
  end
  return t
end

task main()
  var is = ispace(int1d, 11)
  var r = region(is, int)

  for i in is do
    r[i] = i
  end

  var coloring = c.legion_domain_coloring_create()
  var lo0 : int64[1], hi0 : int64[1], lo1 : int64[1], hi1 : int64[1]
  lo0[0], hi0[0], lo1[0], hi1[0] = 0, 5, 6, 10
  c.legion_domain_coloring_color_domain(
    coloring,
    0,
    c.legion_domain_from_rect_1d {
      lo = c.legion_point_1d_t { x = lo0 },
      hi = c.legion_point_1d_t { x = hi0 } }) -- inclusive
  c.legion_domain_coloring_color_domain(
    coloring,
    1,
    c.legion_domain_from_rect_1d {
      lo = c.legion_point_1d_t { x = lo1 },
      hi = c.legion_point_1d_t { x = hi1 } }) -- inclusive
  var p = partition(disjoint, r, coloring)
  c.legion_domain_coloring_destroy(coloring)

  regentlib.assert(sum(is, r) == 55, "test failed")
  inc(p[0].ispace, p[0], 1)
  regentlib.assert(sum(is, r) == 61, "test failed")
  inc(p[1].ispace, p[1], 10)
  regentlib.assert(sum(is, r) == 111, "test failed")
end
regentlib.start(main)
