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

task inc(r : region(int), y : int)
where reads writes(r) do
  for x in r do
    @x += y
  end
end

task main()
  var r = region(ispace(ptr, 5), int)
  fill(r, 0)

  var p = partition(equal, r, ispace(int1d, 3))

  for i = 0, 3 do
    inc(p[i], 10)
  end
end
-- main:getast():printpretty(true)
regentlib.start(main)
