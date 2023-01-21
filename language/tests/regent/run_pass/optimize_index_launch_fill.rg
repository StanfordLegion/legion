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

task main()
  var r = region(ispace(int1d, 10), int64)
  var p = partition(equal, r, ispace(int1d, 2))
  __demand(__index_launch)
  for i = 0, 2 do
    fill((p[i]), int32(123))
  end
  -- Force inline mapping to materialize the fill.
  regentlib.assert(r[0] == 123, "test failed")
end
regentlib.start(main)
