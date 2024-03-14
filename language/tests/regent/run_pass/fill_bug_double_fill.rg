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

-- This test exhibits a second fill when run with inline mapping
-- enabled (with or without composite instances).

task increment(values : region(ispace(int1d), double))
where reads writes(values)
do
  for i in values do
    regentlib.c.printf("values[%d] before  %f\n", i, values[i])
    values[i] += 4000
    regentlib.c.printf("values[%d] after   %f\n", i, values[i])
  end
end

task main()
  var values = region(ispace(int1d, 1), double)
  fill(values, 123)
  increment(partition(equal, values, ispace(int1d, 1))[0])
  for i in values do
    regentlib.c.printf("values[%d] finally %f\n", i, values[i])
  end
  regentlib.assert(values[0] == 4123, "test failed")
end
regentlib.start(main)
