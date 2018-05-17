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
-- type_mismatch_inline2.rg:30: type mismatch in argument 1: expected bool but got int32
--   x, y = f(1), g(1)
--                 ^

import "regent"

__demand(__inline)
task f(x : int) : int return 1 end

__demand(__inline)
task g(x : bool) : bool return true end

task main()
  var x, y = 1, false
  x, y = f(1), g(1)
end
main:compile()
