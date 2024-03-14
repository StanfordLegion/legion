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

task main()
  var x : uint = int1d(5)
  regentlib.assert(x == 5, "test failed")

  var y : regentlib.c.legion_domain_point_t = int2d({2, 3})
  regentlib.assert(int2d(y) == int2d({2, 3}), "test failed")

  var z : regentlib.c.legion_point_2d_t = int2d({4, 5})
  regentlib.assert(z.x[0] == 4, "test failed")
  regentlib.assert(z.x[1] == 5, "test failed")
  regentlib.assert(int2d(z) == int2d({4, 5}), "test failed")
end
regentlib.start(main)
