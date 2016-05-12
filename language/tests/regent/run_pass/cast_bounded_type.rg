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

import "regent"

task off(i : int2d, x : int, y : int)
  return int2d { x = i.x - x, y = i.y - y }
end

task main()
  var is = ispace(int2d, { x = 2, y = 2 })
  for i in is do
    var j : int2d = i
    var k : int2d = off(i, -1, -1) -- Call requires implicit cast.
  end
end
regentlib.start(main)
