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

-- Error Reporting

local ast = require("common/ast")

local report = {}

report.info = function(node, ...)
  -- This should be called with an AST node containing a span. The
  -- span is used to determine the source location of the warning.
  assert(node.span:is(ast.location))

  -- Right now, don't actually do anything on info messages.
end

report.warn = function(node, ...)
  -- This should be called with an AST node containing a span. The
  -- span is used to determine the source location of the warning.
  assert(node.span:is(ast.location))

  io.stderr:write(node.span.source)
  io.stderr:write(":")
  io.stderr:write(tostring(node.span.start.line))
  io.stderr:write(": ")
  io.stderr:write(...)
  io.stderr:write("\n")
end

report.error = function(node, ...)
  -- This should be called with an AST node containing a span. The
  -- span is used to determine the source location of the error.
  assert(node.span:is(ast.location))

  -- The compiler cannot handle running past an error anyway, so just
  -- build the diagnostics object here and don't bother reusing it.
  local diag = terralib.newdiagnostics()
  diag:reporterror(
    {
      filename = node.span.source,
      linenumber = node.span.start.line,
      offset = node.span.start.offset,
    }, ...)
  diag:finishandabortiferrors("Errors reported during typechecking.", 2)
  assert(false) -- unreachable
end

return report
