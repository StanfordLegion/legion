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

__demand(__local, __cuda)
task f(r : region(ispace(int1d), int))
where reads writes(r) do
  for x in r do
    r[x] += 1
  end
end

__demand(__leaf, __cuda)
task call_f(r : region(ispace(int1d), int),
            p : partition(disjoint, r, ispace(int1d)))
where reads writes(r) do
  f(r)
  for i = 0, 2 do
    f(p[i])
  end
end

task main()
  var r = region(ispace(int1d, 10), int)
  var p = partition(equal, r, ispace(int1d, 2))
  fill(r, 0)

  call_f(r, p)

  for x in r do
    regentlib.assert(r[x] == 2, "test failed")
  end
end
regentlib.start(main)
