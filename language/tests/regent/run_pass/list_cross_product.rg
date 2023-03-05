-- Copyright 2023 Stanford University, NVIDIA Corporation
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

task inc(r : region(int), y : int)
where reads writes(r) do
  for x in r do
    @x += y
  end
end

task inc_list(p : regentlib.list(region(int)), y : int)
where reads writes(p) do
  for i = 0, 2 do
    inc(p[i], y)
  end
end

task mul(r : region(int))
where reads writes(r) do
  var t = 1
  for x in r do
    t *= @x
  end
  return t
end

task main()
  var r = region(ispace(ptr, 4), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)

  var colors0 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors0, 0, __raw(x0))
  c.legion_coloring_add_point(colors0, 0, __raw(x1))
  c.legion_coloring_add_point(colors0, 1, __raw(x2))
  c.legion_coloring_add_point(colors0, 1, __raw(x3))
  var part0 = partition(disjoint, r, colors0)
  c.legion_coloring_destroy(colors0)

  var colors1 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors1, 0, __raw(x0))
  c.legion_coloring_add_point(colors1, 1, __raw(x1))
  c.legion_coloring_add_point(colors1, 0, __raw(x2))
  c.legion_coloring_add_point(colors1, 1, __raw(x3))
  var part1 = partition(disjoint, r, colors1)
  c.legion_coloring_destroy(colors1)

  var idx = list_range(0, 2)
  var list0 = list_duplicate_partition(part0, idx)
  var list1 = list_duplicate_partition(part1, idx)

  var prod = list_cross_product(list0, list1)
  var prod_shallow = list_cross_product(list0, list1, true)
  var slice = list_range(1, 2)
  var prod_complete = list_cross_product_complete(
    list0[slice], prod_shallow[slice])

  @x0 = 2
  @x1 = 3
  @x2 = 4
  @x3 = 5

  copy(r, list0)
  copy(r, list1)

  inc_list(prod[0], 10)
  inc_list(prod_complete[0], 200)

  for i = 0, 1 do
    copy((list1[i]), (part1[i]))
  end

  c.printf("x0 %d\n", @x0)
  c.printf("x1 %d\n", @x1)
  c.printf("x2 %d\n", @x2)
  c.printf("x3 %d\n", @x3)
  var t = mul(r)
  c.printf("product of x0..x4: %d\n", t)
  regentlib.assert(t == 36720, "test failed")
end
regentlib.start(main)
