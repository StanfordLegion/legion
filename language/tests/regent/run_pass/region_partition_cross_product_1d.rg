-- Copyright 2023 Stanford University, NVIDIA Corporation
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

-- Test: cross product between two partitions on structured index space (1D).

import "regent"

fspace access_info {
  -- This location is accessed through partition cp[dim1][dim2].
  { dim1 } : int,
  { dim2 } : int,
  -- #times location is accessed.
  count : int,
}

task main()
  var is = ispace(int1d, 6)

  -- Color spaces for partitions.
  var is_a = ispace(int1d, 2)
  var is_b = ispace(int1d, 3)

  var r = region(is, access_info)

  for i in is do
    r[i].count = 0
  end

  var pa = partition(equal, r, is_a)
  var pb = partition(equal, r, is_b)
  var cp = cross_product(pa, pb)
  
  -- Access each location through the cross product.
  for i1 in is_a do
    var p1 = cp[i1]
    for i2 in is_b do
      var p2 = p1[i2]
      for i in p2 do
        p2[i].dim1 = i1
        p2[i].dim2 = i2
        p2[i].count += 1
      end
    end
  end

  -- Verify access info.
  for i1 in is_a do
    var ra = pa[i1]
    for i in ra do
      regentlib.assert(ra[i].dim1 == [int](i1), "access index 1 x doesn't match")
    end
  end
  for i2 in is_b do
    var rb = pb[i2]
    for i in rb do
      regentlib.assert(rb[i].dim2 == [int](i2), "access index 2 x doesn't match")
    end
  end
  for i in is do
    regentlib.assert(r[i].count == 1, "access count doesn't match")
  end
end

regentlib.start(main)

