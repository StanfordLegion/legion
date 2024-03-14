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

local c = regentlib.c

task f(r : region(ispace(int1d), int))
where reads(r), writes(r) do
  for e in r do
    @e = [int](e)
  end
end

task main()
  var r = region(ispace(int1d, 10), int)
  var cs = ispace(int1d, 2)
  var p = partition(equal, r, cs)
  var q = partition(equal, r, cs)
  var cp = cross_product(p, q)
  var cp0 = static_cast(partition(disjoint, r, cs), cp[0])

  fill(r, 123)
  __demand(__index_launch)
  for i in cs do
    f(cp0[i])
  end

  for e in cp[0][0] do
    regentlib.assert(@e == [int](e), "test failed")
  end
  for e in cp[0][1] do
    regentlib.assert(@e == 123, "test failed")
  end
  for e in cp[1][0] do
    regentlib.assert(@e == 123, "test failed")
  end
  for e in cp[1][1] do
    regentlib.assert(@e == 123, "test failed")
  end
end
regentlib.start(main)
