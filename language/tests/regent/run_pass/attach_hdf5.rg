-- Copyright 2016 Stanford University
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
-- []

import "regent"

fspace t {
  a : int32,
  b : int64,
  c : double,
}

local filename = os.tmpname() .. ".hdf"

task main()
  var is = ispace(int3d, {4, 4, 4})
  var r = region(is, t)
  attach(hdf5, r.{a, b, c}, filename, regentlib.file_create)
  acquire(r)
  release(r)
  detach(hdf5, r.{a, b, c})
end
regentlib.start(main)
