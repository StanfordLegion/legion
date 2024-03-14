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

local c = regentlib.c

task test_structured()
  var is = ispace(int2d, {10, 10})
  var r = region(is, int)
  var color_space = ispace(int2d, {2, 2})
  var p = partition(equal, r, color_space)
  for e in r do @e = e.x + e.y end
  for color in p.colors do
    for e in p[color] do
      regentlib.assert(@e == e.x + e.y, "test failed")
    end
  end
end

task main()
  test_structured()
end

regentlib.start(main)
