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

-- This test tests a bug where unsigned bounds to an index launch
-- resulted in an infinite loop.

task f()
end

task main()
  var low : uint = 0
  var high : uint = 0
  __demand(__index_launch)
  for i = low, high do
    f()
  end
end
regentlib.start(main)
