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

-- [["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "9"]]

-- This code has not been optimized and is not high performance.

terra abs(a : double) : double
  if a < 0 then
    return -a
  else
    return a
  end
end

__demand(__parallel)
task saxpy(x : region(ispace(int1d), float), y : region(ispace(int1d), float), a : float)
where
  reads(x, y), writes(y)
do
  __demand(__vectorize)
  for i in x do
    y[i] += a*x[i]
  end
end

task test(n : int)
  var is = ispace(int1d, n)
  var x = region(is, float)
  var y = region(is, float)

  fill(x, 1.0)
  fill(y, 0.0)

  saxpy(x, y, 0.5)
  for i in is do
    regentlib.assert(abs(y[i] - 0.5) < 0.00001, "test failed")
  end
end

task main()
  test(10)
  test(20)
end
regentlib.start(main)
