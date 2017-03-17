-- Copyright 2017 Stanford University
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

fspace t {
  x : int,
}

task f() : int
  var r = region(ispace(ptr, 10), t)

  var p0 = new(ptr(t, r))
  var p1 = new(ptr(t, r))
  var p2 = new(ptr(t, r))
  var p3 = new(ptr(t, r))
  var p4 = new(ptr(t, r))

  p0.x = 5
  p1.x = 40
  p2.x = 300
  p3.x = 2000
  p4.x = 10000

  for p in r do
    p.x = p.x + 100000
  end

  var s = 0
  for p in r do
    s = s + p.x
  end

  return s
end

task main()
  regentlib.assert(f() == 512345, "test failed")
end
regentlib.start(main)
