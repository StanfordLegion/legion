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
-- vectorize_loops10.rg:33: vectorization failed: found a loop-carried dependence
--     a[e.v] = 1
--      ^

import "regent"

fspace fs
{
  v : int,
}

task f(r : region(fs))
where reads writes(r)
do
  var a : int[10]
  __demand(__vectorize)
  for e in r do
    a[e.v] = 1
  end
end
