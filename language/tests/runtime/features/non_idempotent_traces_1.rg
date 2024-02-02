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

-- runs-with:
-- [["-dm:memoize", "-ll:cpu", "2"]]

-- This test is a basic test to see that the runtime can record and
-- replay a non-idempotent trace that only has a single template.

import "regent"

local c = regentlib.c

task reducer(r : region(ispace(int1d), int))
where
  reduces +(r)
do
  for i in r do
    r[i] += 2
  end
end

task reader(r : region(ispace(int1d), int), iter : int)
where
  reads(r)
do
  for x in r do
    regentlib.assert(r[x] == (4 * (iter + 1)), "test failed")
  end
end

task main()
  var is = ispace(int1d, 100)
  var r = region(is, int)
  -- Partition the region into some overlapping pieces.
  var is1 = ispace(int1d, 4)
  var is2 = ispace(int1d, 5)
  var p1 = partition(equal, r, is1)
  var p2 = partition(equal, r, is2)

  var num_loops = 100
  fill(r, 0)

  -- Record a simple non-idempotent trace. One way to do this
  -- is to have the trace produce a reduction that is read by
  -- a task not in the trace. This is because the trace will have
  -- an anti-condition and post-condition of the reduction buffer.
  
  for j = 0, num_loops do
    c.legion_runtime_begin_trace(__runtime(), __context(), 15210, false)
    for i in is2 do
      reducer(p2[i])
    end
    for i in is2 do
      reducer(p2[i])
    end
    c.legion_runtime_end_trace(__runtime(), __context(), 15210)
    for i in is1 do
      reader(p1[i], j)
    end
  end
end

regentlib.start(main)
