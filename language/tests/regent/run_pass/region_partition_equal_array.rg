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

-- FIXME: This needs a shim for Realm before it will run.

import "regent"

struct t {
  value : int,
}

task f()
  var r = region(ispace(int2d, { x = 3, y = 5 }), t)
  var c1 = ispace(int2d, { x = 3, y = 1 })
  var p1 = partition(equal, r, c1)

  for i in c1 do
    var ri = p1[i]
    for x in ri do
      x.value = (1 + i.x) * (1 + i.x)
    end
  end

  var s = 0
  for i in c1 do
    var ri = p1[i]
    for x in ri do
      s += x.value
    end
  end

  return s
end

task g()
  var r = region(ispace(int2d, { x = 3, y = 5 }), t)
  var c2 = ispace(int2d, { x = 2, y = 2 })
  var p2 = partition(equal, r, c2)

  for i in c2 do
    var ri = p2[i]
    for x in ri do
      x.value = (1 + i.x) * (1 + i.y)
    end
  end

  var s = 0
  for i in c2 do
    var ri = p2[i]
    for x in ri do
      s += x.value
    end
  end

  return s
end

task main()
  regentlib.assert(f() == 70, "test failed")
  regentlib.assert(g() == 28, "test failed")
end
regentlib.start(main)
