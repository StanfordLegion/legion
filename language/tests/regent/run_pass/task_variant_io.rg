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
-- [["-ll:io", "1"]]

import "regent"

local launcher = require("std/launcher")
local cmapper = launcher.build_library("task_variant_io")

local c = regentlib.c

-- Right now there's no way to force this to run on an IO processor
-- but with the default kind ranking it should get assigned automatically.
task f()
  var proc =
    c.legion_runtime_get_executing_processor(__runtime(), __context())
  c.printf("executing on processor %llx\n", proc.id)
  regentlib.assert(c.legion_processor_kind(proc) == c.IO_PROC, "test failed")
end
-- This is here to test that there isn't a duplicate registration in
-- the case where the user sets an explicit variant ID.
f:get_primary_variant():set_variant_id(123)

task main()
  f()
end
launcher.launch(main, "task_variant_io", clayout_test.register_mappers, {"-ltask_variant_io"})
