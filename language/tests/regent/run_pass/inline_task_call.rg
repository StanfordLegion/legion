-- Copyright 2015 Stanford University
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

task h(s : region(int), y : ptr(int, s)) : int
where reads(s), writes(s) do
  @y += 100
  return @y
end

__demand(__inline)
task id(x : int) : int
  return x
end

task g() : int
  var r = region(ispace(ptr, 5), int)
  var x = new(ptr(int, r))
  @x = 5
  __demand(__inline, f(r, x))
  __demand(__inline, h(r, x))
  f(r, x)
  for i = 0, 5 do
    h(r, x)
  end
  return __demand(__inline, f(r, x))
end

task main()
  regentlib.assert(g() == 605, "test failed")
end
regentlib.start(main)
