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

-- This tests a crash in conflicting reduction privileges.

local c = regentlib.c

task g2b(s : region(int)) : int
where reads(s) do
  return 5
end

task h2(s : region(int)) : int
where reduces +(s) do
  return 5
end

task h2b(s : region(int)) : int
where reduces *(s) do
  return 5
end

task with_partitions(r1 : region(int), p1_disjoint : partition(disjoint, r1),
                     n : int)
where reads(r1), writes(r1) do

  for i = 0, n do
    g2b(p1_disjoint[i])
  end

  for i = 0, n do
    h2b(p1_disjoint[i])
  end

  for i = 0, n do
    h2(p1_disjoint[i])
  end
end

task main()
  var n = 1
  var r = region(ispace(ptr, n), int)
  var rc = c.legion_coloring_create()
  for i = 0, n do
    c.legion_coloring_ensure_color(rc, i)
  end
  var p_disjoint = partition(disjoint, r, rc)
  var r1 = p_disjoint[0]
  var p1_disjoint = partition(disjoint, r1, rc)
  c.legion_coloring_destroy(rc)

  with_partitions(r1, p1_disjoint, n)
end
regentlib.start(main)
