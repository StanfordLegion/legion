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
-- type_mismatch_import_region3.rg:27: type mismatch in argument 3: expected a logical region handle but got int32
--   var r = __import_region(is, int, x, raw_fids)
--                                    ^

import "regent"

task main()
  var x : int
  var is = ispace(int1d, 5)
  var raw_r : regentlib.c.legion_logical_region_t
  var raw_fids : regentlib.c.legion_field_id_t[1]
  var r = __import_region(is, int, x, raw_fids)
end
