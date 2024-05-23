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

fspace fs {
  a : int,
  b : int,
  c : int,
}

__demand(__local, __leaf)
task f(r : region(ispace(int1d), int), v : int)
where reads writes(r) do
  for x in r do
    r[x] += v
  end
end

__demand(__leaf)
task call_f(r : region(ispace(int1d), fs))
where reads writes(r) do
  f(r.{a}, 1)
  f(r.{b}, 20)
  f(r.{c}, 300)
end

task main()
  var r = region(ispace(int1d, 10), fs)
  fill(r.{a, b, c}, 0)

  call_f(r)

  for x in r do
    regentlib.assert(r[x].a == 1, "test failed")
    regentlib.assert(r[x].b == 20, "test failed")
    regentlib.assert(r[x].c == 300, "test failed")
  end
end
regentlib.start(main)
