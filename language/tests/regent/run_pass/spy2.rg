-- Copyright 2022 Stanford University
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

-- A series of progressively more complicated tests to test Legion Spy capability.

import "regent"

task hello()
  return 1
end

task world(x : int)
  return x + 1
end

task other(x : int)
end

task again(x : int)
end

task main()
  var f = world(hello())
  other(f)
  again(f)
end
regentlib.start(main)