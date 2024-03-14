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

-- fails-with:
-- privilege_field_access_complex1.rg:25: invalid privilege writes($r.x) for dereference of ptr(test(), $r)

import "regent"

fspace test {
  x : complex64,
}

task foo(r : region(test))
  r[0].x.real = 1
end

task main()
  var r = region(ispace(ptr, 5), test)
  foo(r)
end
regentlib.start(main)
