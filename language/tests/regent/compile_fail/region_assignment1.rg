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
-- region_assignment1.rg:40: invalid call missing constraint $x <= $s

import "regent"

task assert_subregion(x : region(int), y : region(int))
where
  x <= y
do
end

task main()
  var s = region(ispace(ptr, 5), int)
  var t = region(ispace(ptr, 5), int)

  s[0] = 1
  t[0] = 2

  var x : region(int)
  if true then
    x = s
  else
    x = t
  end

  assert_subregion(x, s)
end
regentlib.start(main)
