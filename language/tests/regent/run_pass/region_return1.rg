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

import "regent"

task switch(b : bool, s : region(int), t : region(int)) : region(int)
where
  reads(s),
  reads(t)
do
  if b then
    return s
  else
    return t
  end
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

  var r = switch(true, s, t)

  regentlib.assert(x[0] == r[0], "test failed")
  x[0] = 0
  regentlib.assert(r[0] == 0, "test failed")
end
regentlib.start(main)
