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

-- runs-with:
-- [
--   ["-fflow", "1"],
--   ["-fflow", "0"]
-- ]

import "regent"

-- This tests https://github.com/StanfordLegion/legion/issues/727

task f()
  var x = 0
  var y = 0
  x = 123
  y = 2 * 10 * x
  x = 456
  return y
end

task main()
  regentlib.c.printf("%d\n", f())
  regentlib.assert(f() == 2460, "test failed")
end
regentlib.start(main)
