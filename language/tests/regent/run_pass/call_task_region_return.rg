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

__demand(__inline)
task f(s : region(int)) : ptr(int, s)
  return dynamic_cast(ptr(int, s), 0)
end

task g(t : region(int), y : ptr(int, t)) : int
where reads(t) do
  return @y
end

task h() : int
  var r = region(ispace(ptr, 5), int)
  var x = f(r)
  @x = 5
  return g(r, x)
end

task main()
  regentlib.assert(h() == 5, "test failed")
end
regentlib.start(main)
