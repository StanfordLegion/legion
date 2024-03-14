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
-- vectorize_loops7.rg:32: vectorization failed: loop body has a non-scalar if-condition
--    if e.v > 1 then e.v = 0 end
--     ^

import "regent"

fspace fs
{
  v : float,
}

task f(r : region(fs))
where reads writes(r)
do
  __demand(__vectorize)
  for e in r do
    if e.v > 1 then e.v = 0 end
  end
end

regentlib.start(toplevel)
