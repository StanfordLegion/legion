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
-- [["-fflow", "0"]]

-- fails-with:
-- optimize_index_launch_noninterference_cross_product.rg:37: loop optimization failed: argument 2 interferes with argument 1

import "regent"

task foo(r : region(ispace(int1d), int), s : region(ispace(int1d), int))
where reads writes(r, s) do
end

task main()
  var R = region(ispace(int1d, 20), int)
  fill(R, 0)

  var p = partition(equal, R, ispace(int1d, 2))
  var q = partition(equal, R, ispace(int1d, 5))
  var r = partition(equal, R, ispace(int1d, 10))
  var cp = cross_product(p, q, r)

  __demand(__index_launch)
  for i in p.colors do
    foo(cp[i][i][i], p[i])
  end

end
regentlib.start(main)

