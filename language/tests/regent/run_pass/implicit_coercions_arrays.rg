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

task test1d()
  var is = ispace(int1d, 5)
  var r = region(is, int)
  r[3] = 123
end

task test2d()
  var is = ispace(int2d, { x = 2, y = 2 })
  var r = region(is, int)
  r[{ x = 1, y = 1 }] = 123
end

task main()
  test1d()
  test2d()
end
regentlib.start(main)
