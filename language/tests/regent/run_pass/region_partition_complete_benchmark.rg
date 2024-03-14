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

struct t {
  value : int
}

local c = regentlib.c

task test(explicit : bool)
  var rsize = 10
  var r = region(ispace(int2d, {rsize, rsize}), t)
  var t : transform(2, 2)
  t[{0, 0}] = 1
  t[{1, 0}] = 0
  t[{0, 1}] = 0
  t[{1, 1}] = 1
  var e : rect2d = [rect2d] { lo = [int2d] { 0, 0 }, hi = [int2d] { 1, 1 } }
  var csize = 10
  var colors = ispace(int2d, {csize, csize})
  if explicit then
    var p = restrict(aliased, complete, r, t, e, colors)
    c.printf("p is complete %d\n", c.legion_index_partition_is_complete(__runtime(), __raw(p).index_partition))
  else
    var p = restrict(aliased, r, t, e, colors)
    c.printf("p is complete %d\n", c.legion_index_partition_is_complete(__runtime(), __raw(p).index_partition))
  end
end

task main()
  __fence(__execution, __block)
  var t0 = c.legion_get_current_time_in_micros()
  test(false)
  __fence(__execution, __block)
  var tf = c.legion_get_current_time_in_micros()
  c.printf("unspecified %f\n", 1e-6 * (tf - t0))

  __fence(__execution, __block)
  t0 = c.legion_get_current_time_in_micros()
  test(true)
  __fence(__execution, __block)
  tf = c.legion_get_current_time_in_micros()
  c.printf("specified %f\n", 1e-6 * (tf - t0))
end
regentlib.start(main)
