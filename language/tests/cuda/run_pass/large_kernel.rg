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
-- [[ "-ll:gpu", "1", "-ll:fsize", "7000", "-ll:csize", "7000" ]]

import "regent"

local format = require("std/format")

__demand(__cuda)
task inc(r : region(ispace(int1d), int8), v : int8)
where reads writes(r) do
  for e in r do
    r[e] += v
  end
end

task check(r : region(ispace(int1d), int8), v : int8)
where reads(r) do
  var pass = true
  var last_fail_value : int64 = -1
  var last_fail_index : int1d = -1
  var total_failures = 0
  for e in r do
    if last_fail_index >= int1d(0) and r[e] ~= last_fail_value then
      if last_fail_index < e-1 then
        format.println("(repeats another {} times)", e-last_fail_index-1)
      end
      last_fail_index = int1d(-1)
    end
    if r[e] ~= v then
      if r[e] ~= last_fail_value then
        total_failures += 1
        if total_failures <= 10 then
          format.println("expected {} but got: r[{}] = {}", v, e, r[e])
          last_fail_value = r[e]
          last_fail_index = e
        end
      end
      pass = false
    end
  end
  if last_fail_index >= int1d(0) and last_fail_index < r.bounds.hi then
    format.println("(repeats another {} times)", r.bounds.hi-last_fail_index)
  end
  return pass
end

task test_size(size : int64)
  format.println("Running size {} ({.2e})", size, double(size))
  var r = region(ispace(int1d, size), int8)
  fill(r, 1)
  inc(r, 20)
  var ok = check(r, 21)
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
  test_size(7000000000LL)
end
regentlib.start(main)
