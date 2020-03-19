-- Copyright 2020 Stanford University
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

struct t {
  value : int
}

local c = regentlib.c

task f()
  -- create a 1000 x 1000 region
  var rsize = 1000
  var r = region(ispace(int2d, {rsize, rsize}), t)
  -- Make a 2x2 identity matrix for the transform
  var t : transform(2, 2)
  t[{0, 0}] = 1
  t[{1, 0}] = 0
  t[{0, 1}] = 0
  t[{1, 1}] = 1
  -- rectangle
  var e : rect2d = [rect2d] { lo = [int2d] { 0, 0 }, hi = [int2d] { 1, 1 } }
  -- create colorspace for partition
  var csize = 1000
  var colors = ispace(int2d, {csize, csize})
  -- Choose one of the following two lines to compare speed with and without completeness checks
  -- var p = restrict(disjoint, r, t, e, colors)
  var p = restrict(disjoint, complete, r, t, e, colors)
  c.printf("p is complete %d\n", c.legion_index_partition_is_complete(__runtime(), __raw(p).index_partition))
end

task main()
  __fence(__execution, __block)
  var t0 = c.legion_get_current_time_in_micros()
  f()
  __fence(__execution, __block)
  var tf = c.legion_get_current_time_in_micros()
  c.printf("%f\n", 1e-6 * (tf - t0))
end

regentlib.start(main)
