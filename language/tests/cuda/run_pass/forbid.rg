-- Copyright 2023 Stanford University, NVIDIA Corporation
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
-- [["-ll:gpu", "1" ]]

import "regent"

__demand(__cuda)
task init(r : region(ispace(int1d), double),
          p : partition(disjoint, r, ispace(int1d)))
where
  reads writes(r)
do
  __forbid(__cuda)
  for c in p.colors do
    var s = p[c]
    for e in s do
      @e = 10.0
    end
  end
end

task check(r : region(ispace(int1d), double))
where
  reads(r)
do
  for e in r do
    regentlib.assert(@e == 10.0, "test failed")
  end
end

task main()
  var r = region(ispace(int1d, 100), double)
  var p = partition(equal, r, ispace(int1d, 10))

  init(r, p)
  check(r)
end

regentlib.start(main)
