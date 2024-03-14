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

task hello(r : region(int))
  regentlib.c.printf("hello\n")
end

task hello2(r : region(int))
  regentlib.c.printf("hello2\n")
  hello(r)
end

task main()
  var r = region(ispace(ptr, 4), int)
  var p = partition(equal, r, ispace(int1d, 4))
  __demand(__index_launch)
  for i = 0, 4 do
    hello2(p[i])
  end

  -- FIXME: This is needed in nopaint to avoid a race with region deletion
  __fence(__execution)
end
regentlib.start(main)
