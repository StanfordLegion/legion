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

-- runs-with:
-- [["-ll:gpu", "1" ]]

import "regent"

__demand(__parallel, __cuda)
task gpu_task(r : region(ispace(int1d), float))
where reads writes(r)
do
  var x : float = 0
  for e in r do
    x += @e
  end
  return x
end

task main()
  var r = region(ispace(int1d, 1000000), float)

  var cs = ispace(int1d, 2)
  var p_r = partition(equal, r, cs)
  __fence(__execution, __block)
  var ts_start = regentlib.c.legion_get_current_time_in_micros()
  __parallelize_with cs, p_r do
    for i = 0, 10 do
      gpu_task(r)
    end
  end
  __fence(__execution, __block)
  var ts_end = regentlib.c.legion_get_current_time_in_micros()
  regentlib.c.printf("wall-clock time: %lld us\n", ts_end - ts_start)
end

regentlib.start(main)
