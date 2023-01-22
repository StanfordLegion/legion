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
-- invalid_import_region3.rg:26: field size does not match

import "regent"

task main()
  var is = ispace(int1d, 5)
  var r = region(is, int)

  var raw_r = __raw(r)
  var raw_fids = __fields(r)
  var s = __import_region(is, double, raw_r, raw_fids)
end

regentlib.start(main)
