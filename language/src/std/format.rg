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

local function format_value(value, value_type)
  local format_str
  local format_args = terralib.newlist({})
  if std.is_bounded_type(value_type) then
    return format_value(value, value_type.index_type)
  elseif std.is_index_type(value_type)then
    local base_type = value_type.base_type
    if not value_type.fields then
      return format_value(rexpr [base_type](value) end, base_type)
    end

    assert(base_type:isstruct())
    base_type = base_type:getentries()[1].type or base_type:getentries()[1][1]

    format_str = "{"
    for i, field_name in ipairs(value_type.fields) do
      local elt_format, elt_args = format_value(rexpr value.[field_name] end, base_type)
      format_str = format_str .. field_name .. "=" .. elt_format
      if i < value_type.dim then
        format_str = format_str .. ", "
      end
      format_args:insertall(elt_args)
    end
    format_str = format_str .. "}"
  elseif std.is_rect_type(value_type) then
    local index_type = value_type.index_type
    local lo_format, lo_args = format_value(rexpr value.lo end, index_type)
    local hi_format, hi_args = format_value(rexpr value.hi end, index_type)
    format_str = "{lo=" .. lo_format .. ", hi=" .. hi_format .. "}"
    format_args:insertall(lo_args)
    format_args:insertall(hi_args)
  else
    format_str = format_string_mapping[value_type]
    if not format_str then
      report.error(node, "println does not understand how to format a value of type " .. tostring(value_type))
    end
    format_args:insert(value)
  end

  return format_str, format_args
end

local function sanitize(str)
  return string.gsub(str, "%%", "%%%%")
end

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
        local arg_format, arg_args = format_value(args[idx], args[idx]:gettype())
        format_str = format_str .. sanitize(string.sub(msg, last_pos, start-1)) .. arg_format
        format_args:insertall(arg_args)
      end
      idx = idx + 1
      last_pos = stop + 1
      start, stop = next_match()
    end
    format_str = format_str .. sanitize(string.sub(msg, last_pos, -1)) .. "\n"

    if idx-1 ~= #args then
      report.error(node, "println received " .. #args .. " arguments but format string has " .. (idx-1) .. " interpolations")
    end

    return rexpr regentlib.c.printf([format_str], [format_args]) end
  end)

return format
