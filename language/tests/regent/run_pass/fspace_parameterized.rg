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

fspace st(param_r : region(ispace(ptr), int))
{
  x : ptr(int, param_r),
  y : ptr(int, param_r),
}

fspace pst(param_r1 : region(ispace(ptr), int),
           param_r2 : region(ispace(ptr), st(r1)))
{
  p : ptr(st(param_r1), param_r2)[4],
}

task main()
  var r1 = region(ispace(ptr, 2), int)
  var r2 = region(ispace(ptr, 2), st(r1))
  var r3 = region(ispace(ptr, 2), pst(r1, r2))

  var p1_r1 = dynamic_cast(ptr(int, r1), 0)
  var p2_r1 = dynamic_cast(ptr(int, r1), 1)

  @p1_r1, @p2_r1 = 1, 2

  var p1_r2 = dynamic_cast(ptr(st(r1), r2), 0)
  var p2_r2 = dynamic_cast(ptr(st(r1), r2), 1)

  p1_r2.x, p1_r2.y = p1_r1, p2_r1
  p2_r2.y, p2_r2.x = p1_r1, p2_r1

  var p1_r3 = dynamic_cast(ptr(pst(r1, r2), r3), 0)
  var p2_r3 = dynamic_cast(ptr(pst(r1, r2), r3), 1)

  for idx = 0, 4 do
    if idx % 2 == 0 then
      p1_r3.p[idx] = p1_r2
    else
      p1_r3.p[idx] = p2_r2
    end
  end

  @p1_r2.x += 1
  @p1_r2.y -= 1

  regentlib.assert(@(p1_r3.p[0].x) == 2, "test failed")
  regentlib.assert(@(p1_r3.p[0].y) == 1, "test failed")
  regentlib.assert(@(p1_r3.p[1].x) == 1, "test failed")
  regentlib.assert(@(p1_r3.p[1].y) == 2, "test failed")
  regentlib.assert(@(p1_r3.p[2].x) == 2, "test failed")
  regentlib.assert(@(p1_r3.p[2].y) == 1, "test failed")
  regentlib.assert(@(p1_r3.p[3].x) == 1, "test failed")
  regentlib.assert(@(p1_r3.p[3].y) == 2, "test failed")
end

regentlib.start(main)
