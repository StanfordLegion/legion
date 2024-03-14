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

task f0(r : region(int))
where reads(r) do
  var s = 0
  for x in r do
    s += 1
  end
  return s
end

task f1(is : ispace(int1d), r : region(is, int))
where reads(r) do
  var s = 0
  for x in r do
    s += @x
  end
  return s
end

task f2(is : ispace(int2d), r : region(is, int))
where reads(r) do
  var s = 0
  for x in r do
    s += @x
  end
  return s
end

task main()
  var is0 = ispace(ptr, 5)
  var r0 = region(is0, int)
  fill(r0, 0)

  var is1 = ispace(int1d, 7, 2)
  var r1 = region(is1, int)
  fill(r1, 0)

  var is2 = ispace(int2d, { x = 2, y = 2 }, { x = 3, y = 4 })
  var r2 = region(is2, int)
  fill(r2, 0)

  regentlib.assert(f0(r0) == 5, "test failed")

  for i in is1 do
    r1[i] = i
  end
  regentlib.assert(f1(is1, r1) == 35, "test failed")

  for i in is2 do
    r2[i] = 1
  end

  regentlib.assert(f2(is2, r2) == 4, "test failed")
end
regentlib.start(main)
