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

task f(): int
  var x = [ 5 ]
  return x
end

local y = terralib.constant(int, 17)

task g(): int
  var x = [ y ]
  return x
end

task main()
  regentlib.assert(f() == 5, "test failed")
  regentlib.assert(g() == 17, "test failed")
end
regentlib.start(main)
