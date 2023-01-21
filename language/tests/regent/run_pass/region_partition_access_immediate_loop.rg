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

-- This exercises a bug in the Regent compiler when using a subregion
-- which is never stored a variable, but is used as a loop bound.

task main()
  var r = region(ispace(int1d, 10), int)
  var p = partition(equal, r, ispace(int1d, 2))
  for i in p[0] do
    p[0][i] = 123
  end
  for i in p[0] do
    regentlib.assert(p[0][i] == 123, "test failed")
  end
end
regentlib.start(main)
