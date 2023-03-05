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
-- [["-ll:gpu", "1"]]

import "regent"

struct input_fields
{
  f1 : int[3];
  f2 : int[3];
  f3 : int[3];
}

struct output_fields
{
  f1 : int[3];
  f2 : int[3];
}

struct fields
__demand(__cuda)
task nested(r1 : region(ispace(int1d), output_fields),
            r2 : region(ispace(int1d), int[3]),
            r3 : region(ispace(int1d), input_fields))
where
  reads writes(r1), reads(r2, r3)
do
  var lo = [int64](r2.bounds.lo)
  var hi = [int64](r2.bounds.hi) + 1
  for i in r1.ispace do
    for j = lo, hi do
      var v1 = r3[i].f1
      var v2 = r3[i].f2
      var v3 = r3[i].f3
      r1[i].f1 += r2[j] + v1 + v2 + v3
      r1[i].f1 += v1 + v2 + v3
      r1[i].f2 += r2[j] + v1 + v2 + v3
    end
  end
end

task main()
  var r1 = region(ispace(int1d, 10), output_fields)
  var r2 = region(ispace(int1d, 10), int[3])
  var r3 = region(ispace(int1d, 10), input_fields)
  fill(r1.{f1, f2}, array(1, 1, 1))
  fill(r2, array(2, 2, 2))
  fill(r3.{f1, f2, f3}, array(0, 0, 0))

  nested(r1, r2, r3)
  for e in r1 do
    regentlib.assert(r1[e].f1[0] == 21, "test failed")
    regentlib.assert(r1[e].f2[0] == 21, "test failed")
  end
end

regentlib.start(main)

