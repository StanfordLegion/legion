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

-- This test ensures that the compiler scopes regions properly. In
-- general, the compiler caches regions by type, and if it finds a
-- match, it skips codegen the second time. However, caching should
-- respect scope boundaries.

local c = regentlib.c

task f() : int
  var r = region(ispace(ptr, 1), int)
  var p = partition(equal, r, ispace(int1d, 1))

  do
    var r0 = p[0]
    var x = dynamic_cast(ptr(int, r0), 0)
    @x = 1
  end

  do
    var r0 = p[0]
    for x in r0 do @x += 20 end
  end

  do
    var r0 = p[0]
    do
      var r0 = p[0]
      for x in r0 do @x += 300 end
    end

    do
      var r0 = p[0]
      for x in r0 do @x += 4000 end
    end
  end

  if true then
    var r0 = p[0]
    for x in r0 do @x += 50000 end
  else
    var r0 = p[0]
    for x in r0 do @x += 10000 end
  end

  while true do
    var r0 = p[0]
    for x in r0 do @x += 600000 end
    break
  end

  repeat
    var r0 = p[0]
    for x in r0 do @x += 7000000 end
  until true

  for i = 0, 1 do
    var r0 = p[0]
    for x in r0 do @x += 80000000 end
  end

  do
    var r0 = p[0]
    for x in r0 do
      var r0 = p[0]
      for x in r0 do @x += 900000000 end
    end
  end

  do
    var r0 = p[0]
    for x in r0 do return @x end
  end
  return -1
end

task main()
  regentlib.assert(f() == 987654321, "test failed")
end
regentlib.start(main)
