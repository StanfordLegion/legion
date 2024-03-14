-- Copyright 2024 Stanford University, NVIDIA Corporation
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

task foo(r : region(ispace(int1d), int),
         s : region(ispace(int1d), int))
where reads writes(r, s)
do
  for e in r do
    @e += 1
  end
end

task bar(r : region(ispace(int1d), int))
where reads writes(r)
do
  var s = region(ispace(int1d, 5), int)
  for e in r do
    @e += 1
  end
  foo(r, s)
end

task toplevel()
  var r = region(ispace(int1d, 5), int)
  fill(r, 0)
  for i = 0, 10 do
    bar(r)
  end
  for e in r do
    regentlib.assert(@e == 20, "test failed")
  end
end

regentlib.start(toplevel)
