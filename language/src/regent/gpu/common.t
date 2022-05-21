-- Copyright 2022 Stanford University, Los Alamos National Laboratory
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

-- Common Functions for the Regent GPU Code Generator
--
-- IMPORTANT: DO NOT import this file directly, instead see
-- gpu/helper.t for usage.

local base = require("regent/std_base")

local common = {}

function common.generate_slow_atomic(op, typ)
  local cas_type
  if sizeof(typ) == 4 then
    cas_type = uint32
  else
    assert(sizeof(typ) == 8)
    cas_type = uint64
  end
  local terra atomic_op(address : &typ, operand : typ)
    var old : typ = @address
    var assumed : typ
    var new     : typ

    var new_b     : &cas_type = [&cas_type](&new)
    var assumed_b : &cas_type = [&cas_type](&assumed)

    var mask = false
    repeat
      assumed = old
      new     = [base.quote_binary_op(op, assumed, operand)]
      var res = terralib.cmpxchg([&cas_type](address), @assumed_b, @new_b, {success_ordering = "acq_rel", failure_ordering = "monotonic"})
      old     = @[&typ](&(res._0))
      mask    = res._1
    until mask
  end
  atomic_op:setinlined(true)
  return atomic_op
end

function common.generate_atomic_update(op, typ)
  local opname
  if op == "+" then
    if typ:isfloat() then
      opname = "fadd"
    else
      opname = "add"
    end
  elseif op == "min" and typ:isintegral() then
    if typ.signed then
      opname = "min"
    else
      opname = "umin"
    end
  elseif op == "max" and typ:isintegral() then
    if typ.signed then
      opname = "max"
    else
      opname = "umax"
    end
  end
  if opname then
    local terra atomic_op(addr : &typ, value : typ)
      return terralib.atomicrmw(opname, addr, value, {ordering = "monotonic"})
    end
    atomic_op:setinlined(true)
    return atomic_op
  end

  -- If we can't generate a fast atomic, fall back to the slow path
  -- via cmpxchg
  return common.generate_slow_atomic(op, typ)
end

local function count_primitive_fields(ty)
  if ty:isprimitive() or ty:ispointer() then return 1
  elseif ty:isarray() then return count_primitive_fields(ty.type) * ty.N
  else
    assert(ty:isstruct())
    local num_fields = 0
    ty.entries:map(function(entry)
      local field_ty = entry[2] or entry.type
      num_fields = num_fields + count_primitive_fields(field_ty)
    end)
    return num_fields
  end
end

function common.count_arguments(args)
  local num_args = 0
  for i = 1, #args do
    num_args = num_args + count_primitive_fields(args[i].type)
  end
  return num_args
end

function common.generate_arg_setup(output, arr, arg, ty, idx)
  if ty:isprimitive() or ty:ispointer() then
    output:insert(quote [arr][ [idx] ] = &[arg] end)
    return idx + 1
  elseif ty:isarray() then
    for k = 1, ty.N do
      idx = common.generate_arg_setup(output, arr, `([arg][ [k - 1] ]), ty.type, idx)
    end
    return idx
  else
    assert(ty:isstruct())
    ty.entries:map(function(entry)
      local field_name = entry[1] or entry.field
      local field_ty = entry[2] or entry.type
      idx = common.generate_arg_setup(output, arr, `([arg].[field_name]), field_ty, idx)
    end)
    return idx
  end
end

return common
