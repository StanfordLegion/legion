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
  var is1 = ispace(int3d, {2, 1, 1})
  var is2 = ispace(int3d, {2, 1, 1}, {1, 0, 0})
  var is = is1 | is2
  for p in is do
    regentlib.assert(
        p == [int3d]{0, 0, 0} or
        p == [int3d]{1, 0, 0} or
        p == [int3d]{2, 0, 0},
        "test failed")
    regentlib.c.printf("(%d, %d, %d)\n", p.x, p.y, p.z)
  end
end

regentlib.start(main)
