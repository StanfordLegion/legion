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

task f(i : int)
  var point = int1d(regentlib.c.legion_task_get_index_point(__task()))
  regentlib.assert(point == int1d(i), "test failed")
end

task main()
  for t = 0, 3 do
    __forbid(__index_launch)
    for i = 1, 2 do
      f(i)
    end

    var is = ispace(int1d, 1, 2)
    __forbid(__index_launch)
    for i in is do
      f(i)
    end
  end
end
regentlib.start(main)
