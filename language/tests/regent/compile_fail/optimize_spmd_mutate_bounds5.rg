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

-- runs-with:
-- [["-ll:cpu", "4", "-fflow-spmd", "1"]]

-- fails-with:
-- optimize_spmd_mutate_bounds5.rg:45: unable to apply SPMD transformation: leaf loops use inconsistent bounds
--   for i = 0, tstep do
--     ^

-- Test: SPMD should reject code that mutates bounds for inner loops.

import "regent"

task f(r : region(ispace(int1d), int))
where reads writes(r) do
  for x in r do
    @x = 5
  end
end

task main()
  var n = 20
  var nt = 4
  var tstep = 10

  var r = region(ispace(int1d, n), int)
  var part = partition(equal, r, ispace(int1d, nt))

  var lower = 0

  __demand(__spmd)
  for i = 0, tstep do
    nt += 1 -- Mutate loop bound; cannot SPMD.
    for t = lower, nt do f(part[t]) end
    for t = lower, nt do f(part[t]) end
    lower += 1 -- Mutate loop bound; cannot SPMD.
  end
end
regentlib.start(main)

