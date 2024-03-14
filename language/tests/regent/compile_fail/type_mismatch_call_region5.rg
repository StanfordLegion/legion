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

-- fails-with:
-- type_mismatch_call_region5.rg:30: type mismatch in argument 4: expected e($r, $s) but got e($r, $t)
--   f(r, s, t, u)
--   ^

import "regent"

fspace e(a : region(int), b : region(ptr(int, a))) {}

task f(x : region(int), y : region(ptr(int, x)), z : region(ptr(int, x)),
       w : e(x, y))
end

task g(r : region(int), s : region(ptr(int, r)), t : region(ptr(int, r)),
       u : e(r, t))
  f(r, s, t, u)
end
g:compile()
