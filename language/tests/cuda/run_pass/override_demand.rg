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
-- [["-foverride-demand-cuda", "1", "-ll:gpu", "1", "-fflow", "0" ]]

import "regent"

__demand(__cuda)
task init(r : region(ispace(int1d), int))
where writes(r) do
  for e in r do
    r[(e + 1) % r.bounds] = 123
  end
end

task check(r : region(ispace(int1d), int))
where reads(r) do
  for e in r do
    regentlib.assert(@e == 123, "test failed")
  end
end

task main()
  var r = region(ispace(int1d, 100), int)
  init(r)
  check(r)
end

regentlib.start(main)
