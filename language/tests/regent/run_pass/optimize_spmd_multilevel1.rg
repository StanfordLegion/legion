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

-- This tests the SPMD optimization of the compiler with:
--   * multiple top-level partition
--   * multi-level region tree

local format = require("std/format")

task f(r0 : region(int), r1 : region(int))
where reads writes(r0, r1) do
  var a = 0
  for x in r0 do
    a += @x
  end

  var b = 0
  for y in r1 do
    b += @y
  end

  for x in r0 do
    @x = (@x + b) % 1549
  end

  for y in r0 do
    @y = (@y + a) % 1487
  end
end

task g(r0 : region(int))
where reads writes(r0) do
  for x in r0 do
    @x *= 7
  end
end

__demand(__replicable)
task main()
  var grid = region(ispace(ptr, 24), int)

  var LR = partition(equal, grid, ispace(int1d, 2))
  var TB = partition(equal, grid, ispace(int1d, 3))

  var L_all = LR[0]
  var R_all = LR[1]
  var T_all = TB[0]
  var B_all = TB[1]

  var colors = ispace(int1d, 2)
  var L = partition(equal, L_all, colors)
  var R = partition(equal, R_all, colors)
  var T = partition(equal, T_all, colors)
  var B = partition(equal, B_all, colors)

  for x in grid do
    @x = 1000 + int(x)+1 * int(x)+2
  end

  for x in grid do
    format.println("x {} {}", int(x), @x)
  end


  for t = 0, 3 do
    for i in colors do
      f(L[i], R[i])
    end
    for i in colors do
      g(T[i])
      -- f(T[i], B[i]) -- This version needs the workaround for previous consumers.
    end
  end

  format.println("")
  for x in grid do
    format.println("x {} {}", int(x), @x)
  end

  regentlib.assert(grid[0] ==  7420, "test failed")
  regentlib.assert(grid[1] ==  8106, "test failed")
  regentlib.assert(grid[2] ==  8792, "test failed")
  regentlib.assert(grid[3] ==  9478, "test failed")
  regentlib.assert(grid[4] == 10164, "test failed")
  regentlib.assert(grid[5] ==   441, "test failed")
  regentlib.assert(grid[6] ==  2779, "test failed")
  regentlib.assert(grid[7] ==  3465, "test failed")
  regentlib.assert(grid[8] ==   328, "test failed")
end
regentlib.start(main)
