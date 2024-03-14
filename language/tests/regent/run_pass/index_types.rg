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

task named()
  var is = ispace(int2d, { x = 1, y = 3 })
  var t = int2d { x = 10, y = 20 }
  for i in is do
    t += i + { x = 100, y = 200 }
  end
  regentlib.c.printf("named: x = %lld, y = %lld\n", t.x, t.y)
  regentlib.assert(t.x == 310, "test failed")
  regentlib.assert(t.y == 623, "test failed")
end

task positional()
  var is = ispace(int2d, { 1, 3 })
  var t = int2d { 10, 20 }
  for i in is do
    t += i + { 100, 200 }
  end
  regentlib.c.printf("positional: x = %lld, y = %lld\n", t.x, t.y)
  regentlib.assert(t.x == 310, "test failed")
  regentlib.assert(t.y == 623, "test failed")
end

task main()
  named()
  positional()
end
regentlib.start(main)
