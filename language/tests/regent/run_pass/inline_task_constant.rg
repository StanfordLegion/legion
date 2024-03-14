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

__demand(__inline)
task inc(x : int)
  x += 1
  return x
end

task toplevel()
  var i1 = inc(1)
  var i2 = inc(10)
  var c1 = __forbid(__inline, inc(1))
  var c2 = __forbid(__inline, inc(10))
  regentlib.assert(i1 == c1, "test failed")
  regentlib.assert(i2 == c2, "test failed")
  
end

regentlib.start(toplevel)
