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

-- Bishop Logging

local log = {}

log.warn = function(node, ...)
  io.stderr:write(...)
  io.stderr:write("\n")
end

log.error = function(node, ...)
  if node == nil then
    node = { filename = "", linenumber = 0, offset = 0 }
  end

  -- The compiler cannot handle running past an error anyway, so just
  -- build the diagnostics object here and don't bother reusing it.
  local diag = terralib.newdiagnostics()
  diag:reporterror(node.position, ...)
  diag:finishandabortiferrors("Errors reported during typechecking.", 2)
  assert(false) -- unreachable
end

return log
