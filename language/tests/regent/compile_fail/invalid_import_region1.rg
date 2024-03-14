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

-- fails-with:
-- invalid_import_region1.rg:27: $is2 is not the index space of $raw_r

import "regent"

task main()
  var is = ispace(int1d, 5)
  var r = region(is, int)
  var is2 = ispace(int1d, 5)

  var raw_r = __raw(r)
  var raw_fids : regentlib.c.legion_field_id_t[1] = array(100U)
  var s = __import_region(is2, int, raw_r, raw_fids)
end

regentlib.start(main)
