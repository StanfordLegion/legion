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

fspace st(param_r : region(ispace(ptr), int))
{
  x : ptr(int, param_r),
  y : ptr(int, param_r),
}

fspace wrap_st(param_r : region(ispace(ptr), int))
{
  s : st(param_r),
}

fspace pst(param_r1 : region(ispace(ptr), int),
           param_r2 : region(ispace(ptr), wrap_st(param_r1)))
{
  p : ptr(wrap_st(param_r1), param_r2)
}

fspace wrap_pst(param_r1 : region(ispace(ptr), int),
                param_r2 : region(ispace(ptr), wrap_st(param_r1)))
{
  arr : pst(param_r1, param_r2)[2],
}

task main()
  var r1 = region(ispace(ptr, 2), int)
  var r2 = region(ispace(ptr, 2), wrap_st(r1))
  var r3 = region(ispace(ptr, 1), wrap_pst(r1, r2))

  var p1_r1 = dynamic_cast(ptr(int, r1), 0)
  var p2_r1 = dynamic_cast(ptr(int, r1), 1)

  @p1_r1, @p2_r1 = 111, 222

  var p1_r2 = dynamic_cast(ptr(wrap_st(r1), r2), 0)
  var p2_r2 = dynamic_cast(ptr(wrap_st(r1), r2), 1)

  p1_r2.s.x, p1_r2.s.y = p1_r1, p2_r1
  p2_r2.s.y, p2_r2.s.x = p2_r1, p1_r1

  p1_r2.s.x, p1_r2.s.y, p2_r2.s.x, p2_r2.s.y =
    p2_r2.s.y, p2_r2.s.x, p1_r2.s.x, p1_r2.s.y
  regentlib.assert(@p1_r2.s.x == 222, "test failed")
  regentlib.assert(@p1_r2.s.y == 111, "test failed")
  regentlib.assert(@p2_r2.s.x == 111, "test failed")
  regentlib.assert(@p2_r2.s.y == 222, "test failed")

  var p_r3 = dynamic_cast(ptr(wrap_pst(r1, r2), r3), 0)
  var idx0, idx1 = 0, 1
  p_r3.arr[0].p, p_r3.arr[1].p = p1_r2, p2_r2
  var v1_r3 = @p_r3

  p_r3.arr[0], p_r3.arr[1] = p_r3.arr[idx0], p_r3.arr[idx1]
  p_r3.arr[0], p_r3.arr[1] = p_r3.arr[idx1], p_r3.arr[idx0]
  regentlib.assert(@(p_r3.arr[0].p.s.x) == 111, "test failed")
  regentlib.assert(@(p_r3.arr[0].p.s.y) == 222, "test failed")
  regentlib.assert(@(p_r3.arr[1].p.s.x) == 222, "test failed")
  regentlib.assert(@(p_r3.arr[1].p.s.y) == 111, "test failed")
  regentlib.assert(@(v1_r3.arr[0].p.s.x) == 222, "test failed")
  regentlib.assert(@(v1_r3.arr[0].p.s.y) == 111, "test failed")
  regentlib.assert(@(v1_r3.arr[1].p.s.x) == 111, "test failed")
  regentlib.assert(@(v1_r3.arr[1].p.s.y) == 222, "test failed")
  @(p_r3.arr[0].p.s.x), @(p_r3.arr[0].p.s.y) = 1, @(p2_r2.s.x) + @(p1_r2.s.y)
  regentlib.assert(@(p_r3.arr[0].p.s.x) == 1, "test failed")
  regentlib.assert(@(p_r3.arr[0].p.s.y) == 222, "test failed")
  @(p1_r1), @(p_r3.arr[1].p.s.x) = 100, @(p2_r2.s.x) + @(p1_r2.s.y)
  regentlib.assert(@(p1_r1) == 100, "test failed")
  regentlib.assert(@(p_r3.arr[1].p.s.x) == 2, "test failed")
end

regentlib.start(main)
