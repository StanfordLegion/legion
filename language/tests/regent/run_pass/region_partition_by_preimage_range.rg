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

local function f(dim)
  assert(1 <= dim and dim <= 3)
  local tsk
  local index_type
  local rect_type
  local make_struct
  if dim == 1 then
    index_type = regentlib.int1d
    rect_type = regentlib.rect1d
    make_struct = function(x, v) return rexpr x end end
  elseif dim == 2 then
    index_type = regentlib.int2d
    rect_type = regentlib.rect2d
    make_struct = function(x, v) return rexpr { x, v } end end
  elseif dim == 3 then
    index_type = regentlib.int3d
    rect_type = regentlib.rect3d
    make_struct = function(x, v) return rexpr { x, v, v } end end
  end
  task tsk()
    var r = region(ispace(index_type, [make_struct(5, 1)]), int)
    -- pointers in s will be initialized to the first point in r
    var s = region(ispace(int2d, { 5, 1 }), rect_type)
    s[{ 0, 0 }] = [rect_type] { [make_struct(0, 0)], [make_struct(0, 0)] }
    s[{ 1, 0 }] = [rect_type] { [make_struct(1, 0)], [make_struct(1, 0)] }
    s[{ 2, 0 }] = [rect_type] { [make_struct(2, 0)], [make_struct(2, 0)] }
    s[{ 3, 0 }] = [rect_type] { [make_struct(3, 0)], [make_struct(3, 0)] }
    s[{ 4, 0 }] = [rect_type] { [make_struct(4, 0)], [make_struct(4, 0)] }

    var rc = c.legion_domain_point_coloring_create()
    c.legion_domain_point_coloring_color_domain(rc, [int3d] { 0, 0, 0 },
      [rect_type] { [make_struct(0, 0)], [make_struct(0, 0)] })
    c.legion_domain_point_coloring_color_domain(rc, [int3d] { 1, 0, 0 },
      [rect_type] { [make_struct(1, 0)], [make_struct(1, 0)] })
    c.legion_domain_point_coloring_color_domain(rc, [int3d] { 2, 0, 0 },
      [rect_type] { [make_struct(2, 0)], [make_struct(2, 0)] })
    var cs = ispace(int3d, { 3, 1, 1 })
    var p = partition(disjoint, r, rc, cs)
    c.legion_domain_point_coloring_destroy(rc)

    var q = preimage(s, p, s)

    for x in r do
      @x = 1
    end

    for color in cs do
      var si = q[color]
      for y in si do
        var rect = @y
        r[rect.lo] *= color.x + 2
      end
    end

    var t = 0
    for x in r do
      t += @x
    end

    return t
  end
  return tsk
end

task main()
  regentlib.assert([f(1)]() == 11, "test failed")
  regentlib.assert([f(2)]() == 11, "test failed")
  regentlib.assert([f(3)]() == 11, "test failed")
end
regentlib.start(main)
