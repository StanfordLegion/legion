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

local format = require("std/format")
local timing = require("std/timing")

__demand(__leaf)
task leaf_task()
  var x = timing.get_current_time_in_microseconds()
  format.println("in leaf task, current time is {}", x/1.0e6)
end

__demand(__inner)
task inner_task()
  var x = timing.get_current_time_in_microseconds()
  format.println("in inner task, current time is {}", x/1.0e6)
end

__demand(__replicable)
task main()
  var x = timing.get_current_time_in_microseconds()
  -- format.println("in replicable task, current time is {}", x/1.0e6)

  leaf_task()
  inner_task()
end
regentlib.start(main)
