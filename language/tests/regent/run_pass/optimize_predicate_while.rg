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

--runs-with:
-- [[], ["-fpredicate-unroll", "0"], ["-fpredicate-unroll", "1"], ["-fpredicate-unroll", "5"]]

import "regent"

task condition(i : int)
  return i < 10
end

task body(i : int)
  return i + 1
end

task main()
  var i = 0
  __demand(__predicate)
  while condition(i) do
    i = body(i)
  end
  regentlib.assert(i == 10, "test failed")
end
regentlib.start(main)
