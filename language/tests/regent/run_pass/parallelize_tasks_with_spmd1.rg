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

-- [
--  ["-ll:cpu", "4", "-fparallelize-dop", "4",
--   "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"],
--  ["-ll:cpu", "4", "-fbounds-checks", "1",
--   "-fparallelize-dop", "9", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "3"]
-- ]

-- This code has not been optimized and is not high performance.

local abs = regentlib.fabs(double)

__demand(__parallel)
task saxpy(x : region(ispace(int1d), float), y : region(ispace(int1d), float), a : float)
where
  reads(x, y), writes(y)
do
  __demand(__vectorize)
  for i in x do
    y[i] = a*x[i]
  end
end

__demand(__parallel)
task check(y : region(ispace(int1d), float))
where
  reads(y)
do
  for i in y do
    regentlib.assert(abs(@i - 0.5) < 0.00001, "test failed")
  end
end

task test(n : int)
  var is = ispace(int1d, n)
  var x = region(is, float)
  var y = region(is, float)

  fill(x, 1.0)
  fill(y, 0.0)

  __demand(__spmd)
  do
    for idx = 1, 5 do
      saxpy(x, y, 0.5)
      check(y)
    end
  end
end

task main()
  test(100)
  test(200)
end
regentlib.start(main)
