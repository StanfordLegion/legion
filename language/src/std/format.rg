-- Copyright 2020 Stanford University
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

local ast = require("regent/ast")
local report = require("regent/report")
local std = require("regent/std")

local format = {}

local format_string_mapping = {
  [int32] = "%d",
  [int64] = "%lld",
  [uint32] = "%u",
  [uint64] = "%llu",
  [float] = "%f",
  [double] = "%f",
}

format.println = regentlib.macro(
  function(msg, ...)
    local node = msg:getast()

    local args = terralib.newlist({...})
    if not (node.expr:is(ast.specialized.expr.Constant) and
              type(node.expr.value) == "string")
    then
      report.error(node, "println expected first argument to be a format string constant")
    end
    msg = node.expr.value

    local idx = 1
    local last_pos = 1
    local format_str = ""
    local format_args = terralib.newlist()

    local function next_match()
      return string.find(msg, "{}", last_pos)
    end

    local start, stop
    start, stop = next_match()
    while start do
      if idx <= #args then
        local arg_type = args[idx]:gettype()
        local arg_format = format_string_mapping[arg_type]
        if std.is_index_type(arg_type) or std.is_bounded_type(arg_type) then
          local base_type = arg_type
          if std.is_bounded_type(arg_type) then
            base_type = base_type.index_type
          end
          base_type = base_type.base_type
          if base_type:isstruct() then
            base_type = base_type:getentries()[1].type or base_type:getentries()[1][1]
          end
          local elt_format = format_string_mapping[base_type]
          if arg_type.fields then
            arg_format = "{"
            for i, field_name in ipairs(arg_type.fields) do
              arg_format = arg_format .. field_name .. "=" .. elt_format
              if i < arg_type.dim then
                arg_format = arg_format .. ", "
              end
              format_args:insert(rexpr [args[idx]].[field_name] end)
            end
            arg_format = arg_format .. "}"
          else
            format_args:insert(rexpr [base_type]([args[idx]]) end)
            arg_format = elt_format
          end
        elseif not arg_format then
          report.error(node, "println does not understand how to format a value of type " .. tostring(arg_type))
        else
          format_args:insert(args[idx])
        end
        format_str = format_str .. string.sub(msg, last_pos, start-1) .. arg_format
      end
      idx = idx + 1
      last_pos = stop + 1
      start, stop = next_match()
    end
    format_str = format_str .. string.sub(msg, last_pos, -1) .. "\n"

    if idx-1 ~= #args then
      report.error(node, "println received " .. #args .. " arguments but format string has " .. (idx-1) .. " interpolations")
    end

    return rexpr regentlib.c.printf([format_str], [format_args]) end
  end)

return format
