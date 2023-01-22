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
--   ["-ll:cpu", "4", "-fflow-spmd", "0"],
--   ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4"]
-- ]

import "regent"

task inc(r : region(ispace(int1d), int), v : int)
where reads writes(r) do
  for x in r do
    @x += v
  end
end

task main()
  regentlib.c.printf("Main running...\n")
  var num_elts = 1024
  var num_colors = 16
  var colors = ispace(int1d, num_colors)

  var r = region(ispace(int1d, num_elts), int)
  var p = partition(equal, r, colors)

  __demand(__spmd)
  do
    for i in colors do
      inc(p[i], 1)
    end
  end
  regentlib.c.printf("Main complete.\n")
end
regentlib.start(main)
