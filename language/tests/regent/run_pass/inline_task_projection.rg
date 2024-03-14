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
-- [ ["-fflow", "0"] ]

import "regent"

struct vec2
{
  x : double;
  y : double;
}

struct iface
{
  a : double;
  b : double;
}

task f(r : region(iface))
where reads writes(r)
do
  for e in r do
    e.a = 345
    e.b = 543
  end
end

__demand(__inline)
task g(r : region(vec2))
where reads writes(r)
do
  f(r.{a=x, b=y})
end

task main()
  var r = region(ispace(ptr, 5), vec2)
  g(r)
  for e in r do
    regentlib.assert(e.x + e.y == 888, "test failed")
  end
end

regentlib.start(main)
