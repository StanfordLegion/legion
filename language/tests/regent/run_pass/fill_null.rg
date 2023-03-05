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

fspace t(r : region(t(r))) {
  x : ptr(t(r), r),
}

task main()
  var r = region(ispace(ptr, 5), t(r))
  var x = dynamic_cast(ptr(t(r), r), 0)

  fill(r.x, null(ptr(t(r), r)))

  regentlib.assert(isnull(x.x), "test failed")
end
regentlib.start(main)
