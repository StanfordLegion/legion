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

-- runs-with:
-- [
--   ["-fskip-empty-tasks", "0"],
--   ["-fskip-empty-tasks", "1"]
-- ]

import "regent"

-- This tests the compiler optimization to skip empty tasks.

task f(i : int)
  regentlib.c.printf("running f %d\n", i)
end

task g(i : int, r : region(int))
where reads writes(r) do
  regentlib.c.printf("running g %d\n", i)

  for x in r do
    @x += 1
  end
end

task h(i : int, r : region(int), s : region(int))
where reads writes(r, s) do
  regentlib.c.printf("running h %d\n", i)

  for x in r do
    @x += 20
  end
  for x in s do
    @x += 30
  end
end

task main()
  var r = region(ispace(ptr, 3), int)
  var cs = ispace(int1d, 5)
  var p = partition(equal, r, cs)

  var s = region(ispace(ptr, 4), int)
  var q = partition(equal, s, cs)

  fill(r, 0)
  fill(s, 0)

  -- No region arguments, should be called.
  regentlib.c.printf("calling f %d\n", 1234)
  f(1234)

  -- First 3 of 5 tasks should be called.
  for i in cs do
    regentlib.c.printf("calling g %d\n", i)
    g(i, p[i])
  end

  -- First 4 of 5 tasks should be called.
  for i in cs do
    regentlib.c.printf("calling h %d\n", i)
    h(i, p[i], q[i])
  end

  regentlib.assert(r[0] == 21, "test failed")
  regentlib.assert(r[1] == 21, "test failed")
  regentlib.assert(r[2] == 21, "test failed")

  regentlib.assert(s[0] == 30, "test failed")
  regentlib.assert(s[1] == 30, "test failed")
  regentlib.assert(s[2] == 30, "test failed")
  regentlib.assert(s[3] == 30, "test failed")
end
regentlib.start(main)
