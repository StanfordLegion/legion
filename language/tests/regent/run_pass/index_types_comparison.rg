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

local c = regentlib.c

task test_2d()
  var r = region(ispace(int2d, {2, 2}), int)
  for p1 in r do
    for p2 in r do
      var i1 : int2d = p1
      var i2 : int2d = p2
      regentlib.assert((i1 == i2) or (i1 ~= i2), "test failed in 2d case (index type)")
      regentlib.assert((p1 == p2) or (p1 ~= p2), "test failed in 2d case (bounded type)")
    end
  end
end
--test_2d:getdefinition():printpretty(false)

task test_3d()
  var r = region(ispace(int3d, {2, 2, 2}), int)
  for p1 in r do
    for p2 in r do
      var i1 : int3d = p1
      var i2 : int3d = p2
      regentlib.assert((i1 == i2) or (i1 ~= i2), "test failed in 3d case (index type)")
      regentlib.assert((p1 == p2) or (p1 ~= p2), "test failed in 3d case (bounded type)")
    end
  end
end
--test_3d:getdefinition():printpretty(false)

task main()
  test_2d()
  test_3d()
end
regentlib.start(main)
