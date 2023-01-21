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
-- privilege_unpack2.rg:33: invalid privileges in argument 1: reads($c)
--   return f(c)
--          ^

import "regent"

fspace k (r : region(int)) {
  s : region(int),
}

task f(m : region(int))
where reads(m) do
end

task g(a : region(int), b : k(a))
where reads(a) do
  var c = b.s
  return f(c)
end
g:compile()
