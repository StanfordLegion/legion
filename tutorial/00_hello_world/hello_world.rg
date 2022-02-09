-- Copyright 2022 Stanford University
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

-- Every Regent program starts with the following line.
import "regent"

-- The outermost scope of a Regent program is a Lua script. This
-- script executes top to bottom. The line below loads the Regent
-- library for formatted output. The module is stored in a Lua
-- variable (declared with "local").

local format = require("std/format")

-- Regent tasks are declared with the task keyword. Tasks are
-- functions that execute their bodies top-to-bottom.
task hello_world()
  -- Print "Hello World!" followed by a new line.
  format.println("Hello World!")
end

-- Execution begins with a main task. The name "main" is arbitrary,
-- but the task must take no arguments.
task main()
  hello_world()
end

-- The following line starts execution. The call will not return.
regentlib.start(main)
