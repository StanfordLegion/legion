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

task f(s : region(int), y : ptr(int, s)) : int
where reads(s) do
  return @y
end

task g(s : region(int), y : ptr(int, s), z : int)
where reads(s), writes(s) do
  @y += z
end

task h(s : region(int), y : ptr(int, s), z : int)
where reads(s), writes(s) do
  @y *= z
end

task k() : int
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)
  var s = region(ispace(ptr, 5), int)
  var y = dynamic_cast(ptr(int, s), 0)

  @x = 100
  @y = 1

  g(s, y, f(r, x))
  h(r, x, 10)
  g(s, y, f(r, x))
  for i = 0, 2 do
    h(s, y, 2)
  end
  return f(s, y)
end

task main()
  regentlib.assert(k() == (1 + 100 + 1000)*2*2, "test failed")
end
regentlib.start(main)
