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

-- runs-with:
-- [["-finner", "0"], ["-finner", "1"]]

import "regent"

-- This tests a bug in mapping optimization where a reigon which is
-- created locally was unmapped outside of its lexical scope.

task g(r : region(ispace(int1d), int))
where reads writes(r) do
end

task f(r : region(ispace(int1d), int), c : int)
where reads writes(r) do
  -- Do something with r to force this task not to be inner, otherwise
  -- all mapping are ignored.
  r[0] = 1234

  for i = 0, c do
    if c > 0 then
      var s = region(ispace(int1d, 2), int)
      g(s)
    end
  end
end

task main()
  var r = region(ispace(int1d, 5), int)
  f(r, 1)
end
regentlib.start(main)
