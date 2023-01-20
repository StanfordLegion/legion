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

import "regent"

local c = regentlib.c

struct t {
  a : int,
  b : int,
  c : int,
}

task main()
  var r = region(ispace(ptr, 5), t)

  __demand(__vectorize)
  for x in r do
    @x = {1, 2, 3}
  end

  for x in r do
    regentlib.assert(x.a == 1, "test failed")
    regentlib.assert(x.b == 2, "test failed")
    regentlib.assert(x.c == 3, "test failed")
  end
end
regentlib.start(main)
