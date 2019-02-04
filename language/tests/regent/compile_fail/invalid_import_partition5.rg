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

-- fails-with:
-- invalid_import_partition5.rg:38: cannot import a handle that is already imported

import "regent"

local c = regentlib.c

terra create_partition(runtime : c.legion_runtime_t,
                       context : c.legion_context_t,
                       lr : c.legion_logical_region_t,
                       cs : c.legion_index_space_t)
  var ip = c.legion_index_partition_create_equal(runtime, context,
    lr.index_space, cs, 1, -1)
  return c.legion_logical_partition_create(runtime, context, lr, ip)
end

task main()
  var is = ispace(int1d, 5)
  var r = region(is, int)
  var cs = ispace(int1d, 2)
  var raw_p = create_partition(__runtime(), __context(), __raw(r), __raw(cs))

  var p = __import_partition(disjoint, r, cs, raw_p)
  var q = __import_partition(disjoint, r, cs, raw_p)
end

regentlib.start(main)
