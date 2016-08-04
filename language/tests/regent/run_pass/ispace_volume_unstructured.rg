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

-- Test: retrieving volume of unstructured ispace.

import "regent"

task main()
  var is = ispace(ptr, 5)
  regentlib.assert(is.volume == 0, "volume calculation incorrect")

  var r = region(is, int)
  var x0 = new(ptr(int, r))
  var x1 = new(ptr(int, r))
  var x2 = new(ptr(int, r))
  regentlib.assert(is.volume == 3, "volume calculation incorrect")

  var x3 = new(ptr(int, r))
  var x4 = new(ptr(int, r))
  regentlib.assert(is.volume == 5, "volume calculation incorrect")
end
regentlib.start(main)
