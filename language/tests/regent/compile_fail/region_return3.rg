-- Copyright 2019 Stanford University
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
-- region_return3.rg:38: invalid call missing constraint $s * $t

import "regent"

task assert_disjoint(x : region(int), y : region(int))
where
  x * y
do
end

task constructor(s : region(int)) : region(int)
  if false then
    return s
  else
    return region(ispace(ptr, 5), int)
  end
end

task main()
  var s = region(ispace(ptr, 5), int)
  var t = constructor(s)

  assert_disjoint(s, t)
end
regentlib.start(main)
