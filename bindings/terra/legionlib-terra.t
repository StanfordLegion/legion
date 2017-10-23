-- Copyright 2017 Stanford University
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

local legion_c = terralib.includec("legion/legion_c.h")

-- imports opaque types in legion_c.h to the global table
function legion:import_types()
  for k, v in pairs(legion_c)
    do
      if string.match(k, "legion_.*_t") then
        k = string.gsub(k, "legion_", "")
        rawset(_G, k, v)
      end
    end
end

function legion:register_terra_task_void(task_fn, task_id, proc, single, index, vid, opt, name)
  vid = vid or -1
  opt = opt or { leaf = false, inner = false, idempotent = false }
  name = name or task_fn.name
  local ptr = task_fn:getdefinitions()[1]:getpointer()
  legion_c.legion_runtime_register_task_void(task_id, proc, single, index,
                                             vid, opt, name, ptr)
end

function legion:set_terra_registration_callback(func)
  legion_c.legion_runtime_set_registration_callback(
    func:getdefinitions()[1]:getpointer())
end

