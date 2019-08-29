-- Copyright 2019 Stanford University
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

fspace s {
  a : int,
}

task f(r : region(s))
where
  reduces +(r.a)
do
  r[0].a += 10
end

task g(r : region(s))
where
  reads(r),
  writes(r.a)
do
  f(r)
end

task main()
  var r = region(ispace(int1d, 1), a)
  g(r)
end
regentlib.start(main)
