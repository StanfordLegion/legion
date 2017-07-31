-- Copyright 2017 Stanford University
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

-- runs-with:
-- [
-- ]

-- FIXME: Breaks SPMD transformation
--   ["-ll:cpu", "4", "-fflow-spmd", "1"]

import "regent"

-- This tests the SPMD optimization of the compiler with:
--   * multiple top-level partition
--   * multi-level region tree

task f(r0 : region(int), r1 : region(int))
where reads writes(r0), reads(r1) do
  var a = 0
  for x in r0 do
    a += @x
  end

  var b = 0
  for y in r1 do
    b += @y
  end

  for x in r0 do
    @x = (@x + a + b) % 1549
  end
end

task g(r0 : region(int))
where reads writes(r0) do
  for x in r0 do
    @x *= 7
  end
end

task main()
  var grid = region(ispace(ptr, 48), int)

  var LR = partition(equal, grid, ispace(int1d, 2))
  var TB = partition(equal, grid, ispace(int1d, 3))

  var L = LR[0]
  var R = LR[1]
  var T = TB[0]
  var B = TB[1]

  var colors = ispace(int1d, 4)
  var C = partition(equal, grid, colors)
  var L_leaf = partition(equal, L, colors)
  var R_leaf = partition(equal, R, colors)
  var T_leaf = partition(equal, T, colors)
  var B_leaf = partition(equal, B, colors)

  for x in grid do
    @x = 1000 + int(x)+1 * int(x)+2
  end

  for i = 0, 8 do
    regentlib.c.printf("x %2d %5d\n", i, grid[i])
  end

  __demand(__spmd)
  for t = 0, 3 do
    for i in colors do
      g(C[i])
    end
    for i in colors do
      f(L_leaf[i], R_leaf[i])
    end
    for i in colors do
      f(T_leaf[i], B_leaf[i])
    end
    for i in colors do
      g(C[i])
    end
  end

  regentlib.c.printf("\n")
  for i = 0, 8 do
    regentlib.c.printf("x %2d %5d\n", i, grid[i])
  end

  regentlib.assert(grid[0] ==  9534, "test failed")
  regentlib.assert(grid[1] ==  6286, "test failed")
  regentlib.assert(grid[2] ==  3038, "test failed")
  regentlib.assert(grid[3] == 10633, "test failed")
  regentlib.assert(grid[4] ==  7903, "test failed")
  regentlib.assert(grid[5] ==  4655, "test failed")
  regentlib.assert(grid[6] ==  2520, "test failed")
  regentlib.assert(grid[7] == 10115, "test failed")
  regentlib.assert(grid[8] == 10395, "test failed")
end
regentlib.start(main)
