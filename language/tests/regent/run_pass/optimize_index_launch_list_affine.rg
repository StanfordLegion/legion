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

-- This tests the various loop optimizations supported by the
-- compiler.

task g(r : region(int)) : int
where reads(r), writes(r) do
  return 5
end

task check(r : region(int), v : int)
where reads(r)
do
  for e in r do
    regentlib.assert(@e == v, "test failed")
  end
end

task main()
  var n = 5
  var r = region(ispace(ptr, n), int)
  var p = partition(equal, r, ispace(int1d, 5))

  var s = ispace(int1d, 2)

  __demand(__index_launch)
  for i in s do
    g(p[i+0])
  end

  __demand(__index_launch)
  for i in s do
    g(p[i+1])
  end

  var s1 = ispace(int1d, 2, 1)
  var s2 = ispace(int1d, 1, 3)

  __demand(__index_launch)
  for i in s1 do
    g(p[i-0])
  end

  __demand(__index_launch)
  for i in s1 do
    g(p[i-1])
  end

  __demand(__index_launch)
  for i in s1 do
    fill((p[i-0]), 12345)
  end

  __demand(__index_launch)
  for i in s1 do
    check(p[i-0], 12345)
  end

  __demand(__index_launch)
  for i in s1 do
    fill((p[i-1]), 2468)
  end

  __demand(__index_launch)
  for i in s1 do
    check(p[i-1], 2468)
  end

  __demand(__index_launch)
  for i in s2 do
    check(p[i-1], 12345)
  end
end
regentlib.start(main)
