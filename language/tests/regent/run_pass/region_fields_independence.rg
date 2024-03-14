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

-- This test triggered a bug in RDIR due to a failure to apply the
-- correct type-based checks for region aliasing.

fspace fs1
{
  x : int,
  y : int,
}

fspace fs2
{
  x : int, -- Bug is triggered by using identical field names in these two fspaces.
  y : int,
}

task foo(s : region(ispace(int1d), fs1))
where reads writes(s) do end

task bar(t : region(ispace(int1d), fs2))
where reads writes(t) do end

task test()
  var x = region(ispace(int1d, 10), fs1)
  var y = region(ispace(int1d, 10), fs2)
  var cs = ispace(int1d, 2)
  var p_x = partition(equal, x, cs)
  var p_y = partition(equal, y, cs)

  do -- Bug is triggered by privilege summarization at this block.
    for c in cs do foo(p_x[c]) end
    for c in cs do bar(p_y[c]) end
  end
end

regentlib.start(test)
