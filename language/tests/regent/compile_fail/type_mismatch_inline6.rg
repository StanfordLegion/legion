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
-- type_mismatch_inline6.rg:27: type mismatch in var: expected int32 but got {}
--   var x : int = f(r)
--     ^

import "regent"

task f(r : region(ispace(int1d), int))
where reads writes(r) do end

task main()
  var r = region(ispace(int1d, 5), int)
  var x : int = f(r)
end
