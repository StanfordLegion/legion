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

task f(r : region(int), s : region(int), t : region(ptr(int, r, s)))
where
  reads(r, s, t),
  writes(r, s)
do
  for x in t do
    @@x = 5
  end
end

task main()
  var r = region(ispace(ptr, 5), int)
  var s = region(ispace(ptr, 5), int)
  var t = region(ispace(ptr, 10), ptr(int, r, s))

  fill(r, 0)
  fill(s, 0)
  fill(t, dynamic_cast(ptr(int, r, s), 0))

  f(r, s, t)
end
regentlib.start(main)
