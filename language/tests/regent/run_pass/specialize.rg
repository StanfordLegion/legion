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

x = 5 -- Specialize constant
i = int -- Specialize type

task xp1() : i
  return x + 1
end

-- Inside local scope globals should be hidden
function local_scope()
  local x = 2.5 -- Specialize constant
  local i = double -- Specialize type

  local task xp1() : i
    return x + 1.1
  end

  return xp1
end
xp2 = local_scope()

task main()
  regentlib.assert(xp1() == 6, "test failed")
  regentlib.assert(xp2() == 3.6, "test failed")
end
regentlib.start(main)
