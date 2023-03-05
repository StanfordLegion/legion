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

local SAME_ADDRESS_SPACE = 4 -- (1 << 2)

task f(x : int)
  return x + 1
end
f:set_mapper_id(0) -- default mapper
f:set_mapping_tag_id(SAME_ADDRESS_SPACE)

task main()
  regentlib.assert(f(3) == 4, "test failed")
end
regentlib.start(main)
