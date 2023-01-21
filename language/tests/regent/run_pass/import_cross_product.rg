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

local c = regentlib.c

task main()
  var R = region(ispace(int1d, 40), int)
  for i in R do R[i] = i end

  var p = partition(equal, R, ispace(int1d, 2))
  var q = partition(equal, R, ispace(int1d, 5))
  var r = partition(equal, R, ispace(int1d, 10))
  var partitions = array(
                     __raw(p).index_partition,
                     __raw(q).index_partition,
                     __raw(r).index_partition)

  var colors : c.legion_color_t[3]
  var raw_cp = c.legion_terra_index_cross_product_create_multi(
                 __runtime(), __context(), &partitions[0], &colors[0], 3)

  var cp = __import_cross_product(p, q, r, colors, raw_cp)

  var sum = 0
  for i in cp[0][0][0] do
    sum += i
  end
  regentlib.assert(sum == 6, 'test failed')
end
regentlib.start(main)

