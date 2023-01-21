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

-- fails-with:
-- check_parallelizable_region9.rg:36: vectorization failed: found a loop-carried dependence
--     r[0] += vec2 { x = 0.0, y = 0.0 }
--      ^

import "regent"

struct vec2
{
  x : double,
  y : double,
}

terra vec2.metamethods.__add(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x + b.x, y = a.y + b.y }
end

task f(r : region(ispace(int1d), vec2))
where reads writes(r) do
  __demand(__vectorize)
  for e in r do
    r[0] += vec2 { x = 0.0, y = 0.0 }
  end
end
