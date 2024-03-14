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

-- Operator precedence should be consistent across Lua, Terra and Regent.

import "regent"

-- Lua
function test_lua()
  return true or true and false
end

-- Terra
terra test_terra() : bool
  return true or true and false
end

-- Regent
task main()
  regentlib.assert([test_lua()], "test failed: lua")
  regentlib.assert(test_terra(), "test failed: terra")
  regentlib.assert(true or true and false, "test failed: regent")
end
regentlib.start(main)
