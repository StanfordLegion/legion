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

task main()
  var r = region(ispace(ptr, 5), int)
  var x = null(ptr(int, r))
  regentlib.assert(isnull(x), "test failed")

  var is = ispace(ptr, 5)
  var y = null(ptr(is))
  regentlib.assert(isnull(y), "test failed")

  var z = null(&int)
  regentlib.assert(isnull(z), "test failed")
end
regentlib.start(main)
