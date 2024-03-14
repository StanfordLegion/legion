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

task main()
  var r = region(ispace(ptr, 15), int)
  var cs = ispace(int3d, {2, 1, 3})
  var p = partition(equal, r, cs)

  fill(r, 0)

  for c in cs do
    var rc = p[c]
    for x in rc do
      @x = int(x) + 100*c.x + 1000*c.y + 10000*c.z
      regentlib.c.printf("point %lld color %lld %lld %lld value %d\n", x, c.x, c.y, c.z, @x)
    end
  end

  var t = 0
  for x in r do
    t += @x
  end

  regentlib.assert(t == 151005, "test failed")
end
regentlib.start(main)
