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

-- Test: shallow cross product between lists of structured, 2d ispaces.

import "regent"

local c = regentlib.c

-- Assigns the rectangular domain from `lo` to `hi` to `color` in `coloring`.
terra add_subspace(coloring : c.legion_domain_point_coloring_t,
                   color : int1d, lo : int2d, hi : int2d)
  var rect = c.legion_rect_2d_t {
    lo = lo:to_point(),
    hi = hi:to_point(),
  }
  c.legion_domain_point_coloring_color_domain(
    coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(rect))
end

-- Fills region `r` with value `y`.
task myfill(r : region(ispace(int2d), int), y : int)
where reads writes(r) do
  for x in r do
    @x = y
  end
end

-- For each of the `length` regions of the list `p`, fills it with its index in the list.
task myfill_list(p : regentlib.list(region(ispace(int2d), int)), length : int)
where reads writes(p) do
  for i = 0, length do
    myfill(p[i], i)
  end
end

-- Asserts that the values in `r` match those in `expected`.
task verify(r : region(ispace(int2d), int), expected : int[6][6])
where reads(r) do
  for x = 0, 6 do
    for y = 0, 6 do
      if r[{ x=x, y=y }] ~= expected[x][y] then
        c.printf("actual\n")
        for i = 0, 6 do
          for j = 0, 6 do
            c.printf("%3d", r[{ x=i, y=j }])
          end
          c.printf("\n")
        end

        c.printf("expected\n")
        for i = 0, 6 do
          for j = 0, 6 do
            c.printf("%3d", expected[i][j])
          end
          c.printf("\n")
        end

        c.printf("comparing (%d %d) = actual %d (expected %d)\n", x, y, r[{ x=x, y=y }], expected[x][y])
      end
      regentlib.assert(r[{ x=x, y=y }] == expected[x][y], "value doesn't match")
    end
  end
end

task main()
  var is = ispace(int2d, { x = 6, y = 6 })
  var r = region(is, int)

  --[[
  Make first partition.
    0 0 0 0 1 1
    0 0 0 0 1 1
    0 0 0 0 1 1
    0 0 0 0 1 1
    2 2 2 2 2 2
    2 2 2 2 2 2
  ]]
  var is_a = ispace(int1d, 3)
  var coloring_a = c.legion_domain_point_coloring_create()
  add_subspace(coloring_a, 0, { x=0, y=0 }, { x=3, y=3 })
  add_subspace(coloring_a, 1, { x=0, y=4 }, { x=3, y=5 })
  add_subspace(coloring_a, 2, { x=4, y=0 }, { x=5, y=5 })
  var part_a = partition(disjoint, r, coloring_a, is_a)
  c.legion_domain_point_coloring_destroy(coloring_a)

  --[[
  Make second partition.
    0 0 0 0 0 0
    1 1 1 2 2 2
    1 1 1 2 2 2
    1 1 1 3 3 3
    1 1 1 3 3 3
    1 1 1 3 3 3
  ]]
  var is_b = ispace(int1d, 4)
  var coloring_b = c.legion_domain_point_coloring_create()
  add_subspace(coloring_b, 0, { x=0, y=0 }, { x=0, y=5 })
  add_subspace(coloring_b, 1, { x=1, y=0 }, { x=5, y=2 })
  add_subspace(coloring_b, 2, { x=1, y=3 }, { x=2, y=5 })
  add_subspace(coloring_b, 3, { x=3, y=3 }, { x=5, y=5 })
  var part_b = partition(disjoint, r, coloring_b, is_b)
  c.legion_domain_point_coloring_destroy(coloring_b)

  -- Take shallow cross product and then complete it.
  var lh_list = list_duplicate_partition(part_a, list_range(0, 3))
  var rh_list = list_duplicate_partition(part_b, list_range(0, 4))
  var prod_shallow = list_cross_product(lh_list, rh_list, true)
  var prod_complete = list_cross_product_complete(lh_list, prod_shallow)

  -- Verify `prod_complete[0]`.
  -- We fill the entire region with `-1`, and then fill each region in
  -- the list `prod_complete[0]` with its index in the list.
  fill(r, -1); copy(r, lh_list); copy(r, rh_list)
  myfill_list(prod_complete[0], 4)
  for i = 0, 4 do copy((rh_list[i]), (part_b[i])) end
  verify(r, [ terralib.new(int[6][6], {
    {  0,  0,  0,  0, -1, -1 },
    {  1,  1,  1,  2, -1, -1 },
    {  1,  1,  1,  2, -1, -1 },
    {  1,  1,  1,  3, -1, -1 },
    { -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1 },
  }) ])

  -- Verify `prod_complete[1]`.
  fill(r, -1); copy(r, lh_list); copy(r, rh_list)
  myfill_list(prod_complete[1], 3)
  for i = 0, 4 do copy((rh_list[i]), (part_b[i])) end
  verify(r, [ terralib.new(int[6][6], {
    { -1, -1, -1, -1,  0,  0 },
    { -1, -1, -1, -1,  1,  1 },
    { -1, -1, -1, -1,  1,  1 },
    { -1, -1, -1, -1,  2,  2 },
    { -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1 },
  }) ])

  -- Verify `prod_complete[2]`.
  fill(r, -1); copy(r, lh_list); copy(r, rh_list)
  myfill_list(prod_complete[2], 2)
  for i = 0, 4 do copy((rh_list[i]), (part_b[i])) end
  verify(r, [ terralib.new(int[6][6], {
    { -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1 },
    {  0,  0,  0,  1,  1,  1 },
    {  0,  0,  0,  1,  1,  1 },
  }) ])
end
regentlib.start(main)
