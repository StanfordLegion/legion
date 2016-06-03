-- Copyright 2016 Stanford University
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
-- vectorize_loops13.rg:27: vectorization failed: loop body has a corner case statement not supported for the moment
--     r[p] = [int](p)
--                 ^

import "regent"

task toplevel()
  var is = ispace(int1d, 5)
  var r = region(is, int)
  __demand(__vectorize)
  for p in is do
    r[p] = [int](p)
  end
  var sum = 0
  for e in r do
    sum += @e
  end
  regentlib.assert(sum == 10, "test failed")
end

regentlib.start(toplevel)
