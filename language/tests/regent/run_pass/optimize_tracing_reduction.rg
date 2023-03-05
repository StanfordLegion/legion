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
-- [
--  [ "-dm:memoize", "-ll:cpu", "2" ],
--  [ "-dm:memoize", "-ll:cpu", "2", "-lg:no_fence_elision" ],
--  [ "-dm:memoize", "-ll:cpu", "2", "-lg:no_trace_optimization" ]
-- ]

import "regent"
import "bishop"

mapper
end

task f(r : region(ispace(int1d), int))
where
  reads writes(r)
do
  for e in r do
    @e += 1
  end
end

task g(r : region(ispace(int1d), int))
where
  reduces+(r)
do
  for e in r do
    @e += 1
  end
end

task check(r : region(ispace(int1d), int))
where
  reads(r)
do
  for e in r do
    regentlib.assert(@e == 16, "test failed")
  end
end

task main()
  var r = region(ispace(int1d, 4), int)
  fill(r, 1)
  __demand(__trace)
  for i = 0, 5 do
    f(r)
    g(r)
    f(r)
  end
  check(r)
end

regentlib.start(main, bishoplib.make_entry())
