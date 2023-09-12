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
-- [
--   ["-ll:cpu", "4", "-dm:memoize"]
-- ]

import "regent"

local c = terralib.includecstring([[
#include <stdio.h>
#include <unistd.h>
]])

task f(x : int)
  var t = regentlib.c.legion_get_current_time_in_micros()
  c.printf("f %d (time %.1fs)\n", x, t/1.0e6)

  c.usleep(500000)
end

task g(x : int)
  var t = regentlib.c.legion_get_current_time_in_micros()
  c.printf("g %d (time %.1fs)\n", x, t/1.0e6)

  c.usleep(500000)
end

__demand(__replicable)
task main()
  __demand(__trace)
  for i = 0, 3 do
    f(1)
    __fence(__execution)
    g(2)
    __fence(__execution)
    f(3)
    __fence(__execution)
    g(4)
    __fence(__execution)
  end
end
regentlib.start(main)
