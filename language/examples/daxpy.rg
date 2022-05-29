-- Copyright 2022 Stanford University, NVIDIA Corporation
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

local format = require("std/format")

struct point { x : double, y : double, z : double }

task daxpy(points : region(point), alpha : double) 
where 
  reads(points.{x,y}), 
  writes(points.z)
do 
  for p in points do
    points[p].z = alpha * points[p].x + points[p].y
  end
end

task check(points : region(point))
where
  reads(points.z)
do
  for p in points do
    regentlib.assert(points[p].z == 5.0, "daxpy failure")
  end
end

task main()

  format.println("Running Daxpy...")

  var chunks = 4

  var points = region(ispace(ptr,1024), point)
  var part = partition(equal, points, ispace(int1d, chunks))

  fill(points.x, 1.0)
  fill(points.y, 2.0)

  for i = 0, chunks do
    daxpy(part[i], 3.0)
  end

  for i = 0, chunks do
    check(part[i])
  end

  format.println("Done!")
end

regentlib.start(main)
