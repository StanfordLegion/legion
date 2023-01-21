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

-- runs-with:
-- [["-fflow", "0", "-findex-launch-dynamic", "0"]]

-- fails-with:
-- optimize_index_launch_projection1.rg:46: loop optimization failed: argument 2 interferes with argument 1
--     f(p[i], p[i].{x})
--      ^

import "regent"

local c = regentlib.c

struct fs
{
  x : int;
  y : int;
}

task f(r : region(fs), s : region(int)) : int
where writes(r), reads(s) do
  return 5
end

task main()
  var r = region(ispace(ptr, 10), fs)
  var p = partition(equal, r, ispace(int1d, 2))
  fill(r.{x, y}, 1)

  -- not optimized: arguments interfere with each other
  __demand(__index_launch)
  for i in p.colors do
    f(p[i], p[i].{x})
  end
end
regentlib.start(main)
