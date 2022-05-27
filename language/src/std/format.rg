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

--[[--
Utilities for formatted printing.

This module provides macros for formatted printing that are
integrated into the Regent language.

A basic example if the `println` macro, which prints format string
followed by a newline. Each location where `{}` appears is replaced
by the value of one of the arguments. Arguments are referred to in
sequence; there is currently no facility to refer to an arguments
explicitly by number or by name.

```lua
println("Hello {} {} {} world!", 123, 3.14, "asdf")
```

Note that the types of the arguments are inferred automatically, so
there is no need to use a format specifier, aside from when a
specific style is desired.

Styles and other arguments for formatting are always optional, and
are included inside the `{}`. For example, `{x}` formats an integer
as hexadecimal, and `{.3}` formats a floating point number with 3
decimal places. Examples:

```lua
println("{7x}", 123) -- Format as hex, padded on the left up to 7 spaces
println("{e}", 12.4) -- Format in scientific notation
println{"{09}", 123) -- Format padded on the left by up to 9 zeroes
```

## Format Specifiers

The general form of a format specifier is:

```
{[padding][.precision][style]}
```

Currently the following values are supported for each field:

  * Padding: must match `[0-9]*`. Specifies
    the minimum number of spaces the formatted value must
    occupy. If the formatted value is at least this number of
    spaces (or greater), nothing is done. If the formatted value is
    smaller, then it is padded on the left with spaces (or zeroes,
    if the padding value starts with a leading `0`) to ensure that
    the field as a whole occupies the padding number of spaces.

  * Precision: must be a literal period (`.`) followed by
    `[0-9]+`. Specifies the number of decimal places used to format
    a floating point value.

  * Style: must match `[ex]`. Each letter specifies a different
    style. Note that styles may require specific types. Currently
    the following styles are supported:

      * `e`: The exponential style. Requires a floating point
        value.
      * `x`: The hexadecimal style. Requires an integer value.

@module std.format
]]

import "regent"

local ast = require("regent/ast")
local report = require("regent/report")
local std = require("regent/std")

local format = {}

local terra convert_bool(x : bool)
  if x then
    return "true"
  else
    return "false"
  end
end
convert_bool:setinlined(true)
convert_bool.replicable = true

local format_string_mapping = {
  [int8]      = { [""] =   "d", x =   "x", allow_precision = false },
  [int16]     = { [""] =   "d", x =   "x", allow_precision = false },
  [int32]     = { [""] =   "d", x =   "x", allow_precision = false },
  [int64]     = { [""] = "lld", x = "llx", allow_precision = false },
  [uint8]     = { [""] =   "u", x =   "x", allow_precision = false },
  [uint16]    = { [""] =   "u", x =   "x", allow_precision = false },
  [uint32]    = { [""] =   "u", x =   "x", allow_precision = false },
  [uint64]    = { [""] = "llu", x = "llx", allow_precision = false },
  [float]     = { [""] =   "f", e =   "e", allow_precision = true },
  [double]    = { [""] =   "f", e =   "e", allow_precision = true },
  [rawstring] = { [""] =   "s",            allow_precision = false },
  [bool]      = { [""] =   "s",            convert = convert_bool },
}

local function format_string(macro_name, node, value, value_type, modifiers)
  local format_str = format_string_mapping[value_type]
  if not format_str then
    report.error(value:getast().expr, macro_name .. " does not understand how to format a value of type " .. tostring(value_type))
  end
  if string.len(modifiers.precision) > 0 and not format_str.allow_precision then
    report.error(node, macro_name .. " does not support precision specifier on a value of type " .. tostring(value_type))
  end
  local convert = format_str.convert
  format_str = format_str[modifiers.style]
  if not format_str then
    report.error(node, macro_name .. " does not support format style " .. modifiers.style .. " on a value of type " .. tostring(value_type))
  end
  if string.len(modifiers.precision) > 0 then
    format_str = "." .. modifiers.precision  .. format_str
  end
  return "%" .. modifiers.padding .. format_str, convert
end

local function format_value(macro_name, node, value, value_type, modifiers)
  local format_str
  local format_args = terralib.newlist()
  if std.is_bounded_type(value_type) then
    return format_value(macro_name, node, value, value_type.index_type, modifiers)
  elseif std.is_index_type(value_type)then
    local base_type = value_type.base_type
    if not value_type.fields then
      return format_value(macro_name, node, rexpr [base_type](value) end, base_type, modifiers)
    end

    assert(base_type:isstruct())
    base_type = base_type:getentries()[1].type or base_type:getentries()[1][1]

    format_str = "{"
    for i, field_name in ipairs(value_type.fields) do
      local elt_format, elt_args = format_value(macro_name, node, rexpr value.[field_name] end, base_type, modifiers)
      format_str = format_str .. field_name .. "=" .. elt_format
      if i < value_type.dim then
        format_str = format_str .. ", "
      end
      format_args:insertall(elt_args)
    end
    format_str = format_str .. "}"
  elseif std.is_rect_type(value_type) then
    local index_type = value_type.index_type
    local lo_format, lo_args = format_value(macro_name, node, rexpr value.lo end, index_type, modifiers)
    local hi_format, hi_args = format_value(macro_name, node, rexpr value.hi end, index_type, modifiers)
    format_str = "{lo=" .. lo_format .. ", hi=" .. hi_format .. "}"
    format_args:insertall(lo_args)
    format_args:insertall(hi_args)
  elseif std.is_string(value_type) then
    return format_value(macro_name, node, rexpr [rawstring](value) end, rawstring, modifiers)
  else
    local convert
    format_str, convert = format_string(macro_name, node, value, value_type, modifiers)
    if convert then
      value = rexpr convert(value) end
    end
    format_args:insert(value)
  end

  return format_str, format_args
end

local function parse_modifiers(macro_name, node, str)
  local padding, point, precision, style = string.match(str, "^([0-9]*)([.]?)([0-9]*)([a-z]?)$")
  if not (style == "x" or style == "e" or style == "") then
    report.error(node, macro_name .. " does not support the format style " .. style)
  end
  if (string.len(point) > 0) ~= (string.len(precision) > 0) then
    report.error(node, macro_name .. " expected precision following . in format string")
  end
  return {padding = padding, precision = precision, style = style}
end

local function sanitize(str)
  return string.gsub(str, "%%", "%%%%")
end

local function format_arguments(macro_name, msg, args)
  local node = msg:getast()

  if not (node.expr:is(ast.specialized.expr.Constant) and
            type(node.expr.value) == "string")
  then
    report.error(node, macro_name .. " expected first argument to be a format string constant")
  end
  msg = node.expr.value

  local idx = 1
  local last_pos = 1
  local format_str = ""
  local format_args = terralib.newlist()

  local function next_match()
    return string.find(msg, "{[0-9]*[.]?[0-9]*[a-z]?}", last_pos)
  end

  local start, stop
  start, stop = next_match()
  while start do
    if idx <= #args then
      local modifiers = parse_modifiers(macro_name, node, string.sub(msg, start+1, stop-1))
      local arg_format, arg_args = format_value(macro_name, node, args[idx], args[idx]:gettype(), modifiers)
      format_str = format_str .. sanitize(string.sub(msg, last_pos, start-1)) .. arg_format
      format_args:insertall(arg_args)
    end
    idx = idx + 1
    last_pos = stop + 1
    start, stop = next_match()
  end
  format_str = format_str .. sanitize(string.sub(msg, last_pos, -1))

  if idx-1 ~= #args then
    report.error(node, macro_name .. " received " .. #args .. " arguments but format string has " .. (idx-1) .. " interpolations")
  end

  return format_str, format_args
end

--- Print formatted string (no automatic newline).
-- @param msg Format string. Must be a literal constant.
-- @param ... Arguments.
format.print = regentlib.macro(
  function(msg, ...)
    local args = terralib.newlist({...})
    local format_str, format_args = format_arguments("println", msg, args)

    return rexpr regentlib.c.printf(format_str, format_args) end
  end)

--- Print formatted string (with automatic newline).
-- @param msg Format string. Must be a literal constant.
-- @param ... Arguments.
format.println = regentlib.macro(
  function(msg, ...)
    local args = terralib.newlist({...})
    local format_str, format_args = format_arguments("println", msg, args)

    return rexpr regentlib.c.printf([format_str .. "\n"], format_args) end
  end)

--- Print formatted string to stream (no automatic newline).
-- @param stream Destination stream of type `FILE *`.
-- @param msg Format string. Must be a literal constant.
-- @param ... Arguments.
format.fprint = regentlib.macro(
  function(stream, msg, ...)
    local args = terralib.newlist({...})
    local format_str, format_args = format_arguments("fprintln", msg, args)

    return rexpr regentlib.c.fprintf(stream, format_str, format_args) end
  end)

--- Print formatted string to stream (with automatic newline).
-- @param stream Destination stream of type `FILE *`.
-- @param msg Format string. Must be a literal constant.
-- @param ... Arguments.
format.fprintln = regentlib.macro(
  function(stream, msg, ...)
    local args = terralib.newlist({...})
    local format_str, format_args = format_arguments("fprintln", msg, args)

    return rexpr regentlib.c.fprintf(stream, [format_str .. "\n"], format_args) end
  end)

--- Print formatted string to buffer (no automatic newline).
-- @param s Destination buffer of type `rawstring`.
-- @param n Maximum number of bytes to write, including a terminating `NUL` character.
-- @param msg Format string. Must be a literal constant.
-- @param ... Arguments.
format.snprint = regentlib.macro(
  function(s, n, msg, ...)
    local args = terralib.newlist({...})
    local format_str, format_args = format_arguments("fprintln", msg, args)

    return rexpr regentlib.c.snprintf(s, n, format_str, format_args) end
  end)

--- Print formatted string to buffer (with automatic newline).
-- @param s Destination buffer of type `rawstring`.
-- @param n Maximum number of bytes to write, including a terminating `NUL` character.
-- @param msg Format string. Must be a literal constant.
-- @param ... Arguments.
format.snprintln = regentlib.macro(
  function(s, n, msg, ...)
    local args = terralib.newlist({...})
    local format_str, format_args = format_arguments("fprintln", msg, args)

    return rexpr regentlib.c.snprintf(s, n, [format_str .. "\n"], format_args) end
  end)

return format
