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

-- fails-with:
-- invalid_import_ispace2.rg:26: cannot import a subspace

import "regent"

task main()
  var is = ispace(int1d, 10)
  var r = region(is, int)
  var p = partition(equal, r, is)
  var s = p[0]
  var raw_is = __raw(s.ispace)
  var is_from_handle = __import_ispace(int1d, raw_is)
end

regentlib.start(main)
