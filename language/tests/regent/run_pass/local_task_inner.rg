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

import "regent"

task f(r : region(ispace(int1d), int), v : int)
where reads writes(r) do
  for x in r do
    r[x] += v
  end
end

__demand(__local)
task call_f(r : region(ispace(int1d), int), v : int)
where reads writes(r) do
  f(r, v + 1)
end

__demand(__inner)
task call_f2(r : region(ispace(int1d), int))
where reads writes(r) do
  call_f(r, 20)
end

task main()
  var r = region(ispace(int1d, 10), int)
  fill(r, 0)

  call_f2(r)

  for x in r do
    regentlib.assert(r[x] == 21, "test failed")
  end
end
regentlib.start(main)
