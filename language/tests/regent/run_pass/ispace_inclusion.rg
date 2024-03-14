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

local c = regentlib.c

task test(num : int64)
  var is = ispace(ptr, num)
  var agg = true


  var ts_start = [int64](c.legion_get_current_time_in_micros())
  for i = 0L, num do
    agg = agg and (ptr(i) <= is)
  end
  var ts_end = [int64](c.legion_get_current_time_in_micros())
  c.printf("num %8ld, time: %8ld us, time per check: %.2f ns\n", num, ts_end - ts_start,
      (ts_end - ts_start) * 1000.0 / num)

  return agg
end

task main()
  var is = ispace(ptr, 5)

  for i = 0, 5 do
    regentlib.assert(ptr(i) <= is, "test failed")
  end

  for i = 5, 10 do
    regentlib.assert(not (ptr(i) <= is), "test failed")
  end

  var x = 100
  while x <= 1000000L do
    test(x)
    x *= 10
  end
end

regentlib.start(main)
