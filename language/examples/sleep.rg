-- Copyright 2022 Stanford University
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

local c = terralib.includecstring([[
#include <stdio.h>
#include <unistd.h>
]])

task sleep(x : int)
  var start = regentlib.c.legion_get_current_time_in_micros()
  c.printf("subtask %d starting sleep at %llu\n", x, start)
  c.usleep(100000)
  var stop = regentlib.c.legion_get_current_time_in_micros()
  c.printf("subtask %d stopping sleep at %llu, total %llu\n", x, stop, stop - start)
end

task main()
  var start = regentlib.c.legion_get_current_time_in_micros()
  c.printf("main starting to launch subtasks at %llu\n", start)
  for i = 0, 4 do
    sleep(i)
  end
  var stop = regentlib.c.legion_get_current_time_in_micros()
  c.printf("main finished launching subtasks at %llu\n", stop)
end
regentlib.start(main)
