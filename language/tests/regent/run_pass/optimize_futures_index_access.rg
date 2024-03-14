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

task t()
  var c : int[1]
  c[0] = 1
  return c
end

task main()
  var a = t()
  regentlib.assert(a[0] == 1, "test failed")
  a[0] = 2
  regentlib.assert(a[0] == 2, "test failed")
end
regentlib.start(main)
