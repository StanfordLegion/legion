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

import "regent"

local launcher = require("std/launcher")
clegion_interop = launcher.build_library("legion_interop")

struct s {
  a : int32,
  b : int32,
  c : int32,
  d : int32,
}

extern task f(r : region(s), x : regentlib.future(float))
where
  -- Note: With the manual calling convention, these will exactly
  -- correspond to the region requirements passed to the task---so no
  -- grouping or collation of privileges.
  reads(r.{a, b}),
  reads(r.c),
  reads writes(r.d)
end
f:set_task_id(clegion_interop.TID_F)
f:set_calling_convention(regentlib.convention.manual())

task main()
  var r = region(ispace(ptr, 5), s)
  fill(r.{a, b, c, d}, 1.0)

  f(r, 2.0)
end
regentlib.start(main, clegion_interop.register_tasks)
