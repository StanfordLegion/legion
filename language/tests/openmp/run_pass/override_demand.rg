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
-- [["-fopenmp", "1", "-foverride-demand-openmp", "1", "-ll:ocpu", "1", "-ll:othr", "4", "-ll:cpu", "0", "-ll:okindhack", "-fflow", "0" ]]

import "regent"

task init(r : region(ispace(int1d), int))
where writes(r) do
  __demand(__openmp)
  for e in r do
    r[(e + 1) % r.bounds] = 123
  end
end

task check(r : region(ispace(int1d), int))
where reads(r) do
  __demand(__openmp)
  for e in r do
    regentlib.assert(@e == 123, "test failed")
  end
end

task main()
  var r = region(ispace(int1d, 100), int)
  init(r)
  check(r)
end
