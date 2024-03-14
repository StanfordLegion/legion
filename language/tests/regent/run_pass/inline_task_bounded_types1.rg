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

local c = regentlib.c

__demand(__inline)
task foo(r : region(ispace(int1d), int), k : int1d(int, r))
where reads writes(r)
do
  @k += 1
end

__demand(__inline)
task bar(r : region(ispace(int1d), int))
where reads writes(r)
do
  for e in r do foo(r, e) end
end

task main()
  var r = region(ispace(int1d, 4), int)
  fill(r, 1)
  bar(r)
  for e in r do
    regentlib.assert(@e == 2, "test failed")
  end
end

regentlib.start(main)
