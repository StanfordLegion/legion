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

-- fails-with:
-- type_mismatch_fspace4.rg:22: type mismatch in argument 3: expected region(ptr(int32, $rd)) but got region(ptr(int32, $re))
-- fspace s(ra : region(int), rb : region(int), rc : region(ptr(int, ra))) {
--      ^

import "regent"

fspace s(ra : region(int), rb : region(int), rc : region(ptr(int, ra))) {
}

task f(rd : region(int), re : region(int), rf : region(ptr(int, re)), x : s(rd, re, rf))
end
f:compile()
