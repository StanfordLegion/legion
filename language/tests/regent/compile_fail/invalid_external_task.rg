-- Copyright 2021 Stanford University
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

-- fails-with:
-- invalid_external_task.rg:34: dereference in an external task
--     e._0 = 0
--     ^

import "regent"

fspace fs
{
  _0 : float,
  _1 : float,
  _2 : float,
}

__demand(__external)
task init(r : region(ispace(int2d), fs))
where reads writes(r)
do
  for e in r do
    e._0 = 0
  end
end

task toplevel()
  var r = region(ispace(int2d, {3, 3}), fs)
  init(r)
end

regentlib.start(toplevel)
