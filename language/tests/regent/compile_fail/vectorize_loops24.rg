-- Copyright 2017 Stanford University
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
-- vectorize_loops24.rg:30: vectorization failed: loop body has an inner loop
--    for j in is do
--      ^

import "regent"

task foo() end

task toplevel()
  var is = ispace(int1d, 10)

  __demand(__vectorize)
  for i in is do
    __forbid(__parallel)
    for j in is do
      foo()
    end
  end
end

regentlib.start(toplevel)
