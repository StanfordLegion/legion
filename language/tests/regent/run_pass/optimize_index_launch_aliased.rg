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

task foo(r : region(int), s : region(int))
where
  r <= s, reads writes(s)
do
  return 5
end

task toplevel()
  var r = region(ispace(ptr, 10), int)
  var cs = ispace(int1d, 5)
  var p = partition(equal, r, cs)

  var sum = 0
  __demand(__index_launch)
  for c in cs do
    sum += foo(p[c], p[c])
  end
  regentlib.assert(sum == 25, "test failed")
end

regentlib.start(toplevel)
