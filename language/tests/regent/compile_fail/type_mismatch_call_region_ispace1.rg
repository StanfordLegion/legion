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
-- type_mismatch_call_region_ispace1.rg:42: type mismatch in argument 3: expected region(ispace(int1d), float) but got region(ispace(int1d), float)
--     saxpy(px[c].ispace, px[c], py[c], 0.5)
--         ^

import "regent"

task saxpy(is : ispace(int1d), x: region(is, float), y: region(is, float), a: float)
where
  reads(x, y), writes(y)
do
  __demand(__vectorize)
  for i in is do
    y[i] += a*x[i]
  end
end

task test(n: int, np : int)
  var is = ispace(int1d, n)
  var x = region(is, float)
  var y = region(is, float)

  var cs = ispace(int1d, np)
  var px = partition(equal, x, cs)
  var py = partition(equal, y, cs)

  for c in cs do
    saxpy(px[c].ispace, px[c], py[c], 0.5)
  end
end
test:compile()
