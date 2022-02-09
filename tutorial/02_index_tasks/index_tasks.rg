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

local format = require("std/format")

-- Return types may be inferred. The following task returns an int.
task double_of(i : int, x : int)
  format.println("Hello world from task {}!", i)
  return 2*x
end

task main()
  var num_points = 4

  -- Regent automatically converts loops of task calls into index
  -- space launches. The __demand annotation is **NOT** required, but
  -- ensures that the compiler will throw an error if the optimization
  -- fails.
  var total = 0
  __demand(__index_launch)
  for i = 0, num_points do
    total += double_of(i, i + 10)
  end
  regentlib.assert(total == 92, "check failed")
end
regentlib.start(main)
