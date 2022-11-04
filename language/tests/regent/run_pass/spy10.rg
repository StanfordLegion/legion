-- Copyright 2022 Stanford University
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

-- A series of progressively more complicated tests to test Legion Spy capability.

import "regent"

-- This triggers a corner case in Legion Prof+Spy. The "interesting" elements
-- seem to be:
--
--  1. A task launched with no dependencies.
--  2. A fill on a region.
--  3. An index space launch on said region.
--
-- If any element is removed, the case is not triggered. I.e., this is a
-- minimal reproducer.

task hello(r : region(int))
where reads writes(r) do
end

task asdf()
end

task main()
  var r = region(ispace(ptr, 5), int)
  var p = partition(equal, r, ispace(int1d, 2))
  asdf()
  fill(r, 0)
  __fence(__execution)
  __demand(__index_launch)
  for i = 0, 2 do
    hello(p[i])
  end
end
regentlib.start(main)
