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

task reduce(r : region(ispace(int2d), double))
where reduces+(r)
do
  for e in r do @e += 1.0 end
end

task check(r : region(ispace(int2d), double))
where reads(r)
do
  for e in r do
    regentlib.assert(@e == 5.0, "test failed")
  end
end

task main()
  var r = region(ispace(int2d, {4, 4}), double)
  var p = partition(equal, r, ispace(int2d, {2, 2}))
  fill(r, 4.0)
  __forbid(__index_launch)
  for c in p.colors do reduce(p[c]) end
  check(r)
end

regentlib.start(main)
