-- Copyright 2018 Stanford University
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

-- fails-with:
-- 12
-- assertion failed: test passed

import "regent"

-- This is a sanity check to determine that blocks with side effects
-- but no data flow are not reordered by a pass such as RDIR.

local task main()
  do
    regentlib.c.printf("1")
  end
  do
    regentlib.c.printf("2")
  end
  regentlib.c.printf("\n")
  regentlib.assert(false, "test passed")
end
regentlib.start(main)
