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

-- FIXME: this test has a lot commented out because Legion does not
-- support aliased regions in index launches even with simultaneous
-- coherence.

import "regent"

local c = regentlib.c

task f(s0 : region(ispace(int1d), int), s1 : region(ispace(int1d), int))
where
  reads writes simultaneous(s0),
  reads writes simultaneous(s1)
do
end

task g(s0 : region(ispace(int1d), int))
where
  reads writes simultaneous(s0)
do
end

task main()
  var s = region(ispace(int1d, 5), int)

  var cs = ispace(int1d, 5)
  var p1 = partition(equal, s, cs)

  var t = region(cs, rect1d)
  t[0] = rect1d { 0, 1 }
  t[1] = rect1d { 0, 2 }
  t[2] = rect1d { 1, 3 }
  t[3] = rect1d { 2, 4 }
  t[4] = rect1d { 3, 4 }
  var pt = partition(equal, t, cs)
  var p2 = image(s, pt, t)

  -- __demand(__index_launch)
  -- for idx = 0, 1 do
  --   f(p1[0], p1[1]) -- disjoint regions
  -- end

  -- __demand(__index_launch)
  -- for idx = 0, 1 do
  --   f(p2[0], p2[1]) -- aliased regions
  -- end

  -- __demand(__index_launch)
  -- for idx in cs do
  --   g(p2[idx]) -- aliased regions
  -- end

  -- __demand(__index_launch)
  -- for idx in cs do
  --   f(p1[idx], p2[idx]) -- p1 and p2 alias (and p2 self-aliases)
  -- end

  __demand(__index_launch)
  for idx = 0, 1 do -- in cs do
    g(p1[0]) -- same region to all tasks
  end
end
regentlib.start(main)
