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

-- Tests a compiler bug where p.colors would crash inside a leaf task.

__demand(__leaf)
task f(r : region(ispace(int1d), int1d), p : partition(disjoint, r, ispace(int1d)))
  return p.colors.volume
end

task main()
  var c1 = ispace(int1d, 12)
  var r1 = region(c1, int1d)
  var p1 = partition(equal, r1, c1)
  regentlib.assert(f(r1, p1) == 12, "test failed")
end
regentlib.start(main)
