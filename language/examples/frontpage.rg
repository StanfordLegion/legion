-- Copyright 2015 Stanford University
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

fspace point { x : int, y : int, z : int }

task inc(points : region(point))
where reads writes(points.{x, y, z}) do
  for x in points do
    x.{x, y, z} += 1
  end
end

task main()
  var points = region(ispace(ptr, 5), point)
  for i = 0, 5 do new(ptr(point, points)) end
  fill(points.{x, y, z}, 0)

  var colors = ispace(ptr, 3)
  var part = partition(equal, points, colors)

  for i in colors do
    inc(part[i])
  end
end
