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

task f(r : region(ispace(int1d), int))
where writes(r)
do
  r[0] = 10
  return true
end

task main()
  var r = region(ispace(int1d, 1), int)
  fill(r, 1)

  if false and f(r) then end

  if true or f(r) then end

  if (false or true) or f(r) then end

  if (true and false) and (f(r) and f(r)) then end

  regentlib.assert(r[0] == 1, "test failed")
end

regentlib.start(main)
