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

struct s
{
  x : int[10];
}

terra f(x : &int)
  for i = 0, 10 do
    x[i] = i + 1
  end
end

task main()
  var v : s
  for i = 0, 10 do
    v.x[i] = 0
  end
  f([&int](v.x))
  for i = 0, 10 do
    regentlib.assert(v.x[i] == i + 1, "test failed")
  end
end

regentlib.start(main)
