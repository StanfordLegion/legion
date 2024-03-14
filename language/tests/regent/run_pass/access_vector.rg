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

task f(x : vector(int, 4)) : int
  return x[0] + x[3]
end

task g() : int
  var x = vector(1, 20, 300, 4000)
  return f(x)
end

task main()
  regentlib.assert(g() == 4001, "test failed")
end
regentlib.start(main)
