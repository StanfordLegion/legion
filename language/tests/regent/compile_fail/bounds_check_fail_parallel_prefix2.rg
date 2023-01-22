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

-- fails-with:
-- bounds_check_fail_parallel_prefix2.rg:39: parallel prefix operator supports only dense regions

import "regent"

task f()
  var r = region(ispace(int1d, 10), double)
  var s = region(ispace(int1d, 10), double)

  var range = region(ispace(int1d, 3), rect1d)
  range[0] = r.bounds
  range[1] = r.bounds
  range[2] = r.bounds
  var p_range = partition(equal, range, ispace(int1d, 3))

  var p1_r = image(r, p_range, range)
  var p1_s = image(s, p_range, range)

  var p2_r = partition(equal, r, ispace(int1d, 3))
  var p2_s = partition(equal, s, ispace(int1d, 3))

  var d_r = (p1_r - p2_r)[1]
  var d_s = (p1_s - p2_s)[1]

  __parallel_prefix(d_r, d_s, +, 1)
end

regentlib.start(f)
