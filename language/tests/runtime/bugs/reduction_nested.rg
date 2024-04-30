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

-- This test exercised two bugs in the runtime, both associated with
-- nested reduction privileges. One failed with inner optimization,
-- one without. Both are being tested below.

fspace node {
  m : int,
}

task foo_leaf(r : region(node))
where reduces +(r.m) do
  for x in r do x.m += 20 end
end

task foo_non_inner(r : region(node))
where reduces +(r.m) do
  -- for x in r do x.m += 1 end -- Force this to not be inner
  foo_leaf(r)
end

task foo_inner(r : region(node))
where reduces +(r.m) do
  foo_leaf(r)
end

task main()
  var n = 5
  var r = region(ispace(ptr, n), node)
  fill(r.m, 0)
  foo_non_inner(r)
  foo_inner(r)
end
regentlib.start(main)
