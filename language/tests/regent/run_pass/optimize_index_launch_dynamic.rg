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

-- Tests for dynamic index launches.

fspace fs {
  x : int,
  y : int,
  z : int,
}

task f_ro(r : region(fs))
where reads(r) do
end

task f_rw(r : region(fs))
where reads writes(r) do
end

task f_rw3(r : region(fs), s : region(fs), t : region(fs))
where reads writes(r), reads(s, t) do
end

task f_mixed(r : region(fs))
where reads(r.x), reads writes(r.y) do
end

task main()
  var r = region(ispace(ptr, 10), fs)
  var p = partition(equal, r, ispace(int1d, 10))
  fill(r.{x, y, z}, 0)

  -- Always ok, because it's read-only
  __demand(__index_launch)
  for i = 0, 10 do
    f_ro(p[(i+1)%10])
  end

  -- Ok because projection functor is injective
  __demand(__index_launch)
  for i = 0, 10 do
    f_rw(p[(i+1)%10])
  end

  -- Ok because last two arguments are read-only
  __demand(__index_launch)
  for i = 0, 5 do
    f_rw3(p[(i+1)%5], p[5+((i+1)%5)], p[5+((i+1)%5)])
  end

  -- Ok because even with mixed fields, still injective
  __demand(__index_launch)
  for i = 0, 10 do
    f_mixed(p[(i+1)%10])
  end
end

regentlib.start(main)
