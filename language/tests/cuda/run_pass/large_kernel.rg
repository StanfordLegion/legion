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
-- [[ "-ll:gpu", "1" ]]

import "regent"

local format = require("std/format")

__demand(__cuda)
task increment_slot(r : region(ispace(int1d), int64))
where reads writes(r) do
  for e in r do
    r[e%r.bounds] += 1
  end
end

task check(r : region(ispace(int1d), int64))
where reads(r) do
  var total : int64 = 0
  for e in r do
    total += r[e]
  end
  var pass = total == r.volume
  if not pass then
    format.println("expected {} but got {}", r.volume, total)
  end
  return pass
end

task test_size(size : int64)
  -- The exact number of elements doesn't matter, just make it large
  -- enough to reduce contention on atomic operations.
  var buf_size = 16384

  format.println("Running size {} ({.2e})", size, double(size))
  var r = region(ispace(int1d, buf_size), int64)
  fill(r, 0)
  increment_slot(r)
  var ok = check(r)
  regentlib.assert(ok, "test failed")
end

task main()
  -- Run kernels of various sizes.
  test_size(0)
  __fence(__execution, __block)
  test_size(1)
  __fence(__execution, __block)
  test_size(10)
  __fence(__execution, __block)
  test_size(100)
  __fence(__execution, __block)
  test_size(1000)
  __fence(__execution, __block)
  test_size(10000)
  __fence(__execution, __block)
  test_size(100000)
  __fence(__execution, __block)
  test_size(1000000)
  __fence(__execution, __block)
  test_size(10000000)
  __fence(__execution, __block)
  test_size(100000000)
  __fence(__execution, __block)
  test_size(1000000000)
  __fence(__execution, __block)
  test_size(10000000000LL)
end
regentlib.start(main)
