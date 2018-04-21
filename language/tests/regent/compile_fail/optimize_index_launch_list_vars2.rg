-- Copyright 2018 Stanford University
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

-- fails-with:
-- optimize_index_launch_list_vars2.rg:34: loop optimization failed: argument 1 is not provably projectable or invariant
--     f(p[j])
--      ^

import "regent"

task f(r : region(int)) where reads writes(r) do end

terra g(x : int) return x end

task main()
  var cs = ispace(int1d, 4)
  var r = region(ispace(ptr, 5), int)
  var p = partition(equal, r, cs)

  __demand(__parallel)
  for i in cs do
    var j = g(i)
    f(p[j])
  end
end
regentlib.start(main)
