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

-- This file is not meant to be run directly.

-- runs-with:
-- []

import "regent"

require("separate_compilation_common")
require("separate_compilation_tasks_part1_header")
require("separate_compilation_tasks_part2_header")

task main()
  var r = region(ispace(int1d, 5), fs)
  var s = region(ispace(int1d, 10), fs)
  var pr = partition(equal, r, ispace(int1d, 4))
  var ps = partition(equal, s, ispace(int1d, 4))
  fill(r.{x, y, z}, 0)
  fill(s.{x, y, z}, 0)
  for i = 0, 4 do
    my_regent_task(pr[i], 1, 2, true)
  end
  for i = 0, 4 do
    other_regent_task(pr[i], ps[i])
  end
end

-- Save tasks to libseparate_compilation_main.so
local root_dir = arg[0]:match(".*/") or "./"
local separate_compilation_main_h = root_dir .. "separate_compilation_main.h"
local separate_compilation_main_so = root_dir .. "libseparate_compilation_main.so"
regentlib.save_tasks(separate_compilation_main_h, separate_compilation_main_so, nil, nil, nil, nil, false, main)
