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

-- This test is for a regression in the Regent compiler where field
-- access on an index type stripped l-val-ness.

task main()
  var point = int2d { x = 1, y = 2 }
  point.x = 1 -- This should be mutable.
  point.y = 20

  var is = ispace(int2d, { x = 2, y = 2})
  for i in is do
    var point2 = i
    point2.x = 300 -- This should be mutable.
    point2.y = 4000
  end

  regentlib.assert(point.x + point.y == 21, "test failed")
end
regentlib.start(main)
