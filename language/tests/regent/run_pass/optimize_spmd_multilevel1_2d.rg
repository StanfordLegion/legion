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

__demand(__replicable)
task main()
  var grid = region(ispace(int2d, { 4, 4 }), int)
  fill(grid, 0)

  var LR = partition(equal, grid, ispace(int2d, { 2, 1 }))
  var TB = partition(equal, grid, ispace(int2d, { 1, 2 }))

  var L_all = LR[x0y0]
  var R_all = LR[x1y0]
  var T_all = TB[x0y0]
  var B_all = TB[x0y1]

  var colors = ispace(int2d, { 2, 1 }) -- These have to use uniform colors for now.
  var L = partition(equal, L_all, colors)
  var R = partition(equal, R_all, colors)
  var T = partition(equal, T_all, colors)
  var B = partition(equal, B_all, colors)

  for t = 0, 3 do
    for i in colors do
      f(L[i], R[i])
    end
    for i in colors do
      g(T[i])
      -- f(T[i], B[i]) -- This version needs the workaround for previous consumers. But either version hits the barrier index out of bounds issue.
    end
  end
end
regentlib.start(main)
