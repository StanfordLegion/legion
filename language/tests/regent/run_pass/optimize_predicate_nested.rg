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

import "regent"

task condition1(x : int)
  return x % 2 == 0
end

task body1(count : region(ispace(int1d), int), inc : int)
where reads writes(count) do
  for x in count do
    @x += inc
  end
end

task main()
  var count = region(ispace(int1d, 1), int)

  for i = 0, 4 do
    __demand(__predicate)
    if condition1(i) then
      body1(count, 1<<i)
    end
  end

  __demand(__predicate)
  if condition1(1) then
    __demand(__predicate)
    if condition1(2) then
      body1(count, 1<<4)
    end
  end

  __demand(__predicate)
  if condition1(4) then
    __demand(__predicate)
    if condition1(6) then
      body1(count, 1<<5)
    end
  end

  regentlib.assert(count[0] == 37, "test failed")
end
regentlib.start(main)
