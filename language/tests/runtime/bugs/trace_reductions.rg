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

-- This test that runtime precisely computes trace post-conditions
-- and anti-conditions for tasks that use reductions. Prior to the
-- inclusion of this test, the runtime would mistakenly remove reduction
-- views from post-conditions erroneously and overzealously add reduction
-- views to anti-conditions. The test is derived from a simplified version
-- of the circuit application that traces the loops in a rotated order,
-- slightly different than usual.

import "regent"

local c = regentlib.c

fspace fs
{
  field : double;
}

task calculate_new_currents(r : region(ispace(int1d), fs))
where
  reads writes(r)
do
end

task distribute_charge(r : region(ispace(int1d), fs))
where
  reduces +(r.field)
do
end

task update_voltages(r : region(ispace(int1d), fs))
where
  reads writes(r)
do
end


task main()
  var is = ispace(int1d, 100)
  var r = region(is, fs)
  -- Partition the region into some overlapping pieces.
  var is1 = ispace(int1d, 2)
  var is2 = ispace(int1d, 3)
  var p1 = partition(equal, r, is1)
  var p2 = partition(equal, r, is2)
  var num_loops = 100
  fill(r.field, 0.0)

  -- Offset the "normal" sequence of tasks.
  for i in is1 do
    calculate_new_currents(p1[i])
  end
  for i in is2 do
    distribute_charge(p2[i])
  end

  __demand(__trace)
  for j = 0, num_loops do
    for i in is1 do
      update_voltages(p1[i])
    end
    for i in is1 do
      calculate_new_currents(p1[i])
    end
    for i in is2 do
      distribute_charge(p2[i])
    end
  end

  for i in is1 do
    update_voltages(p1[i])
  end
end
regentlib.start(main)