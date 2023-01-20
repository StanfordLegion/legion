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

function make_id(ty) return terra(x : ty) return x end end

local id_int = make_id(int)
local id_double = make_id(double)

task test_unstructured_simple()
  -- disjoint regions of the same fieldspace type (r1 * r2)
  -- disjoint regions of different fieldspace types (r1 * r3)
  var r1 = region(ispace(ptr, 5), int)
  var r2 = region(ispace(ptr, 5), int)
  var r3 = region(ispace(ptr, 5), double)

  var p_r1, p_r2, p_r3 = dynamic_cast(ptr(int, r1), 0), dynamic_cast(ptr(int, r2), 0), dynamic_cast(ptr(double, r3), 0)
  @p_r1, @p_r2, @p_r3 = 1, 2, 3.0
  @p_r1, @p_r2 = @p_r2, [int](@p_r1)
  regentlib.assert(@p_r1 == 2, "test failed")
  regentlib.assert(@p_r2 == 1, "test failed")
  @p_r1, @p_r3 = [int](@p_r3), (@p_r1 + @p_r2) * @p_r3
  regentlib.assert(@p_r1 == 3, "test failed")
  regentlib.assert(@p_r3 == 9.0, "test failed")
  @p_r1, @p_r2 = @p_r2, [int](@p_r3)
  regentlib.assert(@p_r1 == 1, "test failed")
  regentlib.assert(@p_r2 == 9, "test failed")

  @p_r1, @p_r2, @p_r3 = 1, 2, 3.0
  var p2_r1 = dynamic_cast(ptr(int, r1), 1)
  var part1 = partition(equal, r1, ispace(int1d, 2))
  var r10 = part1[0]

  var p_r10 = dynamic_cast(ptr(int, r10), p_r1)
  var p_r1_or_r2 = static_cast(ptr(int, r1, r2), p_r2)

  @p_r10, @p_r2 = @p_r2, id_int(@p_r1 + @p_r2)
  regentlib.assert(@p_r1 == 2, "test failed")
  regentlib.assert(@p_r2 == 3, "test failed")

  @p_r10, @p_r3 = @p_r1, 2 * @p_r3
  regentlib.assert(@p_r1 == 2, "test failed")
  regentlib.assert(@p_r3 == 6.0, "test failed")

  @p_r1_or_r2, @p_r1 = @p_r1, [int](@p_r3 * id_double([double](@p_r2) * 2.5))
  regentlib.assert(@p_r1 == 45, "test failed")
  regentlib.assert(@p_r2 == 2, "test failed")

  var p_r10_old : ptr(int, r10)
  p_r10_old, p_r10, p_r1_or_r2 =
    p_r10, dynamic_cast(ptr(int, r10), p2_r1),
    static_cast(ptr(int, r1, r2), static_cast(ptr(int, r1), p_r10))
  regentlib.assert(
    static_cast(ptr(int, r1), p_r10_old) ==
    dynamic_cast(ptr(int, r1), p_r1_or_r2),
    "test failed")
end

task test_structured_simple()
  var r_1d = region(ispace(int1d, 5), int)
  var r_2d = region(ispace(int2d, {2, 2}), int)

  var idx1, idx2 = 1, 2
  r_1d[0], r_1d[1] = 10, 12
  r_1d[0], r_1d[1] = r_1d[0], r_1d[1]
  r_1d[0], r_1d[1] = r_1d[ [int1d](0) ], r_1d[ [int1d](1) ]
  r_1d[ [int1d](1) ], r_1d[0] = r_1d[ [int1d](0) ], r_1d[ 1 ]
  r_1d[1], r_1d[0] = r_1d[0], r_1d[1]
  regentlib.assert(r_1d[0] == 10, "test failed")
  regentlib.assert(r_1d[1] == 12, "test failed")
  r_1d[1], r_1d[0], r_1d[idx2] = r_1d[0], r_1d[idx1], r_1d[0]
  regentlib.assert(r_1d[0] == 12, "test failed")
  regentlib.assert(r_1d[1] == 10, "test failed")
  regentlib.assert(r_1d[2] == 10, "test failed")

  r_2d[{0, 0}], r_2d[{0, 1}] = 100, 101
  r_2d[{0, idx1}], r_2d[{0, 0}] = r_2d[{0, 0}], r_2d[{0, 1}]
  regentlib.assert(r_2d[{0, 0}] == 101, "test failed")
  regentlib.assert(r_2d[{0, 1}] == 100, "test failed")
end

struct vec2
{
  x : int,
  y : int,
}

vec2.metamethods.__add = terra(a : vec2, b : vec2)
  return vec2 { a.x + b.x, a.y + b.y }
end

vec2.metamethods.__sub = terra(a : vec2, b : vec2)
  return vec2 { a.x - b.x, a.y - b.y }
end

local id_int_ptr = make_id(&int)
local id_vec2 = make_id(vec2)

task test_arrays_and_pointers()
  var arr1 : int[10]
  var arr2 : int[10]
  var p_arr1 : &int = id_int_ptr(arr1)
  for idx = 0, 10 do
    arr1[idx] = idx + 100
    arr2[idx] = idx + 200
  end

  arr2[1], arr1[0] = id_int(arr1[1]), id_int(arr2[0])
  regentlib.assert(arr1[0] == 200, "test failed")
  regentlib.assert(arr2[1] == 101, "test failed")

  p_arr1[0], arr1[1] = p_arr1[1], 2 * [int](arr1[0]) / 2
  regentlib.assert(arr1[0] == 101, "test failed")
  regentlib.assert(arr1[1] == 200, "test failed")

  -- This line introduces a temporary assignment
  -- due to the imprecision in alias analysis
  p_arr1[0], arr2[1] = p_arr1[1], arr2[0]
  regentlib.assert(arr1[0] == 200, "test failed")
  regentlib.assert(arr2[1] == 200, "test failed")

  var arr_2d : vec2[2][2]
  var idx1, idx2 = 0, 1
  arr_2d[0][0], arr_2d[0][1], arr_2d[1][0], arr_2d[1][1] =
    {0, 1}, {2, 3}, {4, 5}, {6, 7}
  arr_2d[0], arr_2d[1] = arr_2d[1], arr_2d[0]
  regentlib.assert(arr_2d[0][0].x == 4, "test failed")
  regentlib.assert(arr_2d[0][0].y == 5, "test failed")
  regentlib.assert(arr_2d[0][1].x == 6, "test failed")
  regentlib.assert(arr_2d[0][1].y == 7, "test failed")
  regentlib.assert(arr_2d[1][0].x == 0, "test failed")
  regentlib.assert(arr_2d[1][0].y == 1, "test failed")
  regentlib.assert(arr_2d[1][1].x == 2, "test failed")
  regentlib.assert(arr_2d[1][1].y == 3, "test failed")
  arr_2d[0][0], arr_2d[0][1] = arr_2d[0][1], arr_2d[0][1] - arr_2d[0][0]
  regentlib.assert(arr_2d[0][0].x == 6, "test failed")
  regentlib.assert(arr_2d[0][0].y == 7, "test failed")
  regentlib.assert(arr_2d[0][1].x == 2, "test failed")
  regentlib.assert(arr_2d[0][1].y == 2, "test failed")

  arr_2d[idx2][0], arr_2d[0][1] =
    arr_2d[0][1], arr_2d[idx2][1] + id_vec2(arr_2d[idx1][0])
  regentlib.assert(arr_2d[1][0].x == 2, "test failed")
  regentlib.assert(arr_2d[1][0].y == 2, "test failed")
  regentlib.assert(arr_2d[0][1].x == 8, "test failed")
  regentlib.assert(arr_2d[0][1].y == 10, "test failed")
end

task main()
  test_unstructured_simple()
  test_structured_simple()
  test_arrays_and_pointers()
end

regentlib.start(main)
