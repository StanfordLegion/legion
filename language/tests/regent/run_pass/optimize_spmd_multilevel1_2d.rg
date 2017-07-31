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
--   ["-ll:cpu", "2", "-fflow-spmd", "1"]
-- ]

import "regent"

-- This tests the SPMD optimization of the compiler with:
--   * multiple top-level partition
--   * multi-level region tree

task f(r0 : region(ispace(int2d), int), r1 : region(ispace(int2d), int))
where reads writes(r0, r1) do
  -- Do nothing, for now
end

task g(r0 : region(ispace(int2d), int))
where reads writes(r0) do
  -- Do nothing, for now
end

local x0y0 = terralib.constant(`int2d { __ptr = regentlib.__int2d { 0, 0 } })
local x0y1 = terralib.constant(`int2d { __ptr = regentlib.__int2d { 0, 1 } })
local x1y0 = terralib.constant(`int2d { __ptr = regentlib.__int2d { 1, 0 } })
local x1y1 = terralib.constant(`int2d { __ptr = regentlib.__int2d { 1, 1 } })

task main()
  var grid = region(ispace(int2d, { 4, 4 }), int)

  var LR = partition(equal, grid, ispace(int2d, { 2, 1 }))
  var TB = partition(equal, grid, ispace(int2d, { 1, 2 }))

  var L = LR[x0y0]
  var R = LR[x1y0]
  var T = TB[x0y0]
  var B = TB[x0y1]

  var colors = ispace(int2d, { 2, 1 }) -- These have to use uniform colors for now.
  var L_leaf = partition(equal, L, colors)
  var R_leaf = partition(equal, R, colors)
  var T_leaf = partition(equal, T, colors)
  var B_leaf = partition(equal, B, colors)

  __demand(__spmd)
  for t = 0, 3 do
    for i in colors do
      f(L_leaf[i], R_leaf[i])
    end
    for i in colors do
      g(T_leaf[i])
      -- f(T_leaf[i], B_leaf[i]) -- This version needs the workaround for previous consumers. But either version hits the barrier index out of bounds issue.
    end
  end
end
regentlib.start(main)
