-- Copyright 2021 Stanford University
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

-- fails-with:
-- optimize_index_launch_noninterference_cross.rg:30: loop optimization failed: argument 2 interferes with argument 1
--        ^

import "regent"

task foo(r: region(ispace(int1d), int), s: region(ispace(int1d), int), t: region(ispace(int1d), int))
where reads writes(r), reads (s, t) do
end

task main()
  var r = region(ispace(int1d, 10), int)
  var p = partition(equal, r, ispace(int1d, 5))

  __demand(__index_launch)
  for i in ispace(int1d, 5) do
    foo(p[i], p[i/1], p[i/2])
  end
end
regentlib.start(main)
