-- Copyright 2016 Stanford University
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

local fields = terralib.newlist({"a", "b", "c"})

local elt = terralib.types.newstruct("elt")
elt.entries = fields:map(function(field) return { field, int } end)

task main()
  var r = region(ispace(ptr, 5), elt)
  new(ptr(elt, r), 3)

  fill(r.[fields], 2)
  for e in r do
    e.[fields] = 1
  end

  for x in r do
    regentlib.assert(x.a + x.b + x.c == 3, "test failed")
  end
end
regentlib.start(main)
