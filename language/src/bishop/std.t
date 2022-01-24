-- Copyright 2022 Stanford University, NVIDIA Corporation
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

local config = require("bishop/config")
local log = require("bishop/log")

local std = {}

std.config, std.args = config.args()

local c = terralib.includecstring ([[
#include "legion.h"
#include "bishop_c.h"
#include <stdio.h>
#include <stdlib.h>
]])

std.c = c

function std.curry(f, a)
  return function(...) return f(a, ...) end
end

function std.curry2(f, a, b)
  return function(...) return f(a, b, ...) end
end

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
register_opaque_type("processor_kind_type")
register_opaque_type("compile_option_type")
register_opaque_type("processor_type")
register_opaque_type("processor_list_type")
register_opaque_type("memory_type")
register_opaque_type("memory_list_type")
register_opaque_type("memory_kind_type")
register_opaque_type("point_type")

function std.is_list_type(type)
  return std.is_processor_list_type(type) or std.is_memory_list_type(type)
end

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
  elseif op == "<" then
    return `([lhs] < [rhs])
  elseif op == "<=" then
    return `([lhs] <= [rhs])
  elseif op == ">" then
    return `([lhs] > [rhs])
  elseif op == ">=" then
    return `([lhs] >= [rhs])
  elseif op == "==" then
    return `([lhs] == [rhs])
  elseif op == "~=" then
    return `([lhs] ~= [rhs])
  elseif op == "and" then
    return `([lhs] and [rhs])
  elseif op == "or" then
    return `([lhs] or [rhs])
  else
    assert(false, "unknown operator " .. tostring(op))
  end
end

function std.make_entry()
  if not rawget(_G, "__bishop_jit_mappers__") then
    log.warn(nil, "No mapper is given. Ignoring register request...")
    return
  end

  local mapper = __bishop_jit_mappers__()
  local num_mapper_impls = #mapper.state_to_mapper_impl + 1
  local mapper_impls =
    terralib.newsymbol(c.bishop_mapper_impl_t[ num_mapper_impls ])
  local num_transitions = #mapper.state_to_transition_impl + 1
  local transitions =
    terralib.newsymbol(c.bishop_transition_fn_t[ num_transitions ])
  local register_body = quote end

  register_body = quote
    [register_body]
    [mapper_impls][0] = c.bishop_mapper_impl_t {
      select_task_options = [c.bishop_select_task_options_fn_t](0),
      slice_task = [c.bishop_slice_task_fn_t](0),
      map_task = [c.bishop_map_task_fn_t](0),
    }
  end
  for i = 1, #mapper.state_to_mapper_impl do
    local mapper_impl = mapper.state_to_mapper_impl[i]
    register_body = quote
      [register_body]
      [mapper_impls][i] = c.bishop_mapper_impl_t {
        select_task_options = [mapper_impl.select_task_options],
        slice_task = [mapper_impl.slice_task],
        map_task = [mapper_impl.map_task],
      }
    end
  end

  for i = 0, #mapper.state_to_transition_impl do
    local transition = mapper.state_to_transition_impl[i]
    register_body = quote
      [register_body]
      [transitions][i] = [transition]
    end
  end

  local terra register()
    var num_mapper_impls = [num_mapper_impls]
    var num_transitions = [num_transitions]
    var [mapper_impls]
    var [transitions]
    [register_body]
    c.register_bishop_mappers([mapper_impls], num_mapper_impls,
                              [transitions], num_transitions,
                              [mapper.mapper_init])
  end

  return register
end

return std
