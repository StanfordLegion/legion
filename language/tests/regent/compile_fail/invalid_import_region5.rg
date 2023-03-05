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
-- invalid_import_region5.rg:39: cannot import a handle that is already imported

import "regent"

local c = regentlib.c

task create_logical_region()
  var is = c.legion_index_space_create(__runtime(), __context(), 1)
  var fs = c.legion_field_space_create(__runtime(), __context())
  var alloc = c.legion_field_allocator_create(__runtime(), __context(), fs)
  c.legion_field_allocator_allocate_field(alloc, [sizeof(int)], 123)
  c.legion_field_allocator_destroy(alloc)
  var lr = c.legion_logical_region_create(__runtime(), __context(), is, fs, false)
  return lr
end

task main()
  var raw_r = create_logical_region()
  var raw_is = raw_r.index_space
  var raw_fids : c.legion_field_id_t[1] = array(123U)

  var is = __import_ispace(int1d, raw_is)
  var r = __import_region(is, int, raw_r, raw_fids)
  var r_again = __import_region(is, int, raw_r, raw_fids)
end

regentlib.start(main)
