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

-- Load the Regent language definition.
import "regent"

-- Define a simple struct to be used later.
struct point {
  x : float,
  y : float,
}

-- Tasks are the fundamental unit of parallelism in Regent. Here, we
-- define 4 tasks. Ignore the task bodies for the moment; the behavior
-- of each task is fully described by its declaration. Note that each
-- declaration says what the task will read or write.
task a(points : region(point)) where writes(points) do end
task b(points : region(point)) where reads writes(points.x) do end
task c(points : region(point)) where reads writes(points.y) do end
task d(points : region(point)) where reads(points) do end

-- Execution typically begins at a main task. Regent code obeys
-- traditional sequential semantics, so read the code top-to-bottom as
-- usual.
task main()
  -- Create a region (like an array) with room for 5 elements.
  var points = region(ispace(ptr, 5), points)
  new(ptr(point, points), 5) -- Allocate the elements.

  -- Partition the region into subregions. Each subregion is a view
  -- onto a subset of the data of the parent.
  var part = partition(equal, points, ispace(int1d, 3))

  -- Launch tasks a, b, c, and d.
  a(points)
  for i = 0, 3 do
    b(part[i])
  end
  c(points)
  for i = 0, 3 do
    d(points)
  end
end

-- Begin execution of main.
regentlib.start(main)
