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
-- [["-foverride-demand-cuda", "1", "-ll:gpu", "1", "-fflow", "0" ]]

import "regent"

struct vec2
{
  x : int,
  y : int,
}

terra vec2.metamethods.__add(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x + b.x, y = a.y + b.y }
end

__demand(__cuda)
task init(r : region(ispace(int1d), vec2))
where reads writes(r) do
  for e in r do
    @e = vec2 { x = 0, y = 0 }
  end
end

__demand(__cuda)
task red(r : region(ispace(int1d), vec2))
where reads writes(r) do
  for e in r do
    r[(e + 1) % r.bounds] += vec2 { x = 123, y = 456 }
  end
end

task check(r : region(ispace(int1d), vec2))
where reads writes(r) do
  for e in r do
    regentlib.assert(r[e].x == 123, "test failed")
    regentlib.assert(r[e].y == 456, "test failed")
  end
end

task main()
  var r = region(ispace(int1d, 100), vec2)
  init(r)
  red(r)
  check(r)
end

regentlib.start(main)
