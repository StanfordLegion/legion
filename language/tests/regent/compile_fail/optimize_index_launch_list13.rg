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

-- fails-with:
-- optimize_index_launch_list13.rg:34: loop optimization failed: fill value is not provably invariant
--     fill((p[i]), f())
--                   ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

terra f() return 1 end

task g()
  var cs = ispace(int1d, 5)
  var r = region(cs, int)
  var p = partition(equal, r, cs)
  -- not optimized: argument is not invariant
  __demand(__index_launch)
  for i in cs do
    fill((p[i]), f())
  end
end
