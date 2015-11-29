-- Copyright 2015 Stanford University, NVIDIA Corporation
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


-- Bishop Standard Libary

local c = terralib.includecstring [[
#include "legion_c.h"
#include "bishop_c.h"
#include <stdio.h>
#include <stdlib.h>
]]

local std = {}

terra std.assert(x : bool, message : rawstring)
  if not x then
    var stderr = c.fdopen(2, "w")
    c.fprintf(stderr, "assertion failed: %s\n", message)
    -- Just because it's stderr doesn't mean it's unbuffered...
    c.fflush(stderr)
    c.abort()
  end
end

std.symbol_count = 0

function std.newsymbol()
  local new_var = "__var" .. tostring(std.symbol_count)
  std.symbol_count = std.symbol_count + 1
  return new_var
end

function std.quote_unary_op(op, rhs)
  if op == "-" then
    return `(-[rhs])
  else
    assert(false, "unknown operator " .. tostring(op))
  end
end

local function register_opaque_type(keyword)
  local flag = "is_" .. keyword
  local tbl = {}
  tbl[flag] = true
  setmetatable(tbl, tbl)
  tbl.__index = tbl
  tbl.__tostring = function(self) return keyword end

  std[keyword] = tbl
  std[flag] = function(t) return rawget(t, flag) end
end

register_opaque_type("isa_type")
register_opaque_type("compile_option_type")
register_opaque_type("processor_type")
register_opaque_type("memory_type")
register_opaque_type("memory_kind_type")
register_opaque_type("point_type")

function std.quote_binary_op(op, lhs, rhs)
  if op == "*" then
    return `([lhs] * [rhs])
  elseif op == "/" then
    return `([lhs] / [rhs])
  elseif op == "%" then
    return `([lhs] % [rhs])
  elseif op == "+" then
    return `([lhs] + [rhs])
  elseif op == "-" then
    return `([lhs] - [rhs])
  else
    assert(false, "unknown operator " .. tostring(op))
  end
end

function std.register_bishop_mappers()
  local rules = terralib.newsymbol(&c.bishop_rule_t)
  local register_body = quote end

  for i = 1, #__bishop__ do
    local rule = __bishop__[i]
    register_body = quote
      [register_body];
      [rules][ [i - 1] ] = c.bishop_rule_t {
        select_task_options =
          [c.bishop_task_callback_fn_t]([rule.select_task_options]),
        pre_map_task =
          [c.bishop_region_callback_fn_t]([rule.pre_map_task]),
        select_task_variant =
          [c.bishop_task_callback_fn_t]([rule.select_task_variant]),
        map_task =
          [c.bishop_region_callback_fn_t]([rule.map_task]),
      }
    end
  end

  local terra register()
    var num_rules = [#__bishop__]
    var [rules] : c.bishop_rule_t[#__bishop__]
    [register_body]
    c.register_bishop_mappers([rules], num_rules)
  end

  register()
end

return std
