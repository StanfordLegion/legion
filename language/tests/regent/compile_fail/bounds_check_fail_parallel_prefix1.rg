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
-- bounds_check_fail_parallel_prefix1.rg:23: the source and the target of a parallel prefix operator must have the same size

import "regent"

task f()
  var r = region(ispace(int1d, 5), double)
  var s = region(ispace(int1d, 6), double)
  __parallel_prefix(r, s, +, 1)
end

regentlib.start(f)
