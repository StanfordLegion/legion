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
    var r = region(ispace(index_type, [make_struct(4, 1)]), int)
    var x0 = dynamic_cast(index_type(int, r), [make_struct(0, 0)])
    var x1 = dynamic_cast(index_type(int, r), [make_struct(1, 0)])
    var x2 = dynamic_cast(index_type(int, r), [make_struct(2, 0)])
    var x3 = dynamic_cast(index_type(int, r), [make_struct(3, 0)])
    var s = region(ispace(index_type, [make_struct(4, 1)]), rect_type)
    var y0 = dynamic_cast(index_type(rect_type, s), [make_struct(0, 0)])
    var y1 = dynamic_cast(index_type(rect_type, s), [make_struct(1, 0)])
    var y2 = dynamic_cast(index_type(rect_type, s), [make_struct(2, 0)])
    var y3 = dynamic_cast(index_type(rect_type, s), [make_struct(3, 0)])

    @y0 = { x0, x0 }
    @y1 = { x1, x1 }
    @y2 = { x2, x2 }
    @y3 = { x3, x3 }

    var sc = c.legion_domain_coloring_create()
    c.legion_domain_coloring_color_domain(sc, 0, [rect_type] { y0, y1 })
    c.legion_domain_coloring_color_domain(sc, 1, [rect_type] { y2, y2 })
    c.legion_domain_coloring_color_domain(sc, 2, [rect_type] { y3, y3 })
    var p = partition(disjoint, s, sc)
    c.legion_domain_coloring_destroy(sc)

    var q = image(r, p, s)

    for x in r do
      @x = 1
    end

    for i = 0, 3 do
      var ri = q[i]
      for x in ri do
        @x *= i + 2
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
