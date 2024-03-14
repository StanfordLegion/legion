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

fspace fs
{
  field : int8;
  payload : double[1678];
}

task init(r : region(ispace(int1d), fs))
where
  reads writes(r)
do
end

task compute(r : region(ispace(int1d), fs))
where
  reads(r)
do
end

task toplevel()
  var r = region(ispace(int1d, 9998), fs)

  var is = ispace(int1d, 2)
  var p = partition(equal, r, is)

  init(r)
  for c in is do
    compute(p[c])
  end
end

regentlib.start(toplevel)
