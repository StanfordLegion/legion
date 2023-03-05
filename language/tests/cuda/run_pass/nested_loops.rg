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
-- [["-ll:gpu", "1"]]

import "regent"

__demand(__cuda)
task nested(r1 : region(ispace(int1d), int),
            r2 : region(ispace(int1d), int))
where
  reads(r1, r2)
do
  var sum : int = 0
  for i in r1.ispace do
    for j in r2.ispace do
      sum += r1[i] + r2[i]
    end
  end
  return sum
end

task main()
  var r1 = region(ispace(int1d, 10), int)
  var r2 = region(ispace(int1d, 10), int)
  fill(r1, 1)
  fill(r2, 2)

  var sum = nested(r1, r2)
  regentlib.assert(sum == 300, "test failed")
end

regentlib.start(main)

