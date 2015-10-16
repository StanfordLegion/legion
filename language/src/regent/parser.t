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

-- Legion Parser

local parsing = require("parsing")
local ast = require("regent/ast")
local std = require("regent/std")

local parser = {}

function parser.expr_prefix(p)
  local start = ast.save(p)
  if p:nextif("(") then
    local expr = p:expr()
    p:expect(")")
    return expr

  elseif p:nextif("[") then
    local expr = p:luaexpr()
    p:expect("]")
    return ast.unspecialized.ExprEscape {
      expr = expr,
      span = ast.span(start, p),
    }

  elseif p:matches(p.name) then
    local name = p:next(p.name).value
    p:ref(name)
    return ast.unspecialized.ExprID {
      name = name,
      span = ast.span(start, p),
    }

  else
    p:error("unexpected token in expression")
  end
end

function parser.field(p)
  local start = ast.save(p)
  if p:matches(p.name) and p:lookahead("=") then
    local name = p:next(p.name).value
    p:expect("=")
    local value = p:expr()
    return ast.unspecialized.ExprCtorRecField {
      name_expr = function(env) return name end,
      value = value,
      span = ast.span(start, p),
    }

  elseif p:nextif("[") then
    local name_expr = p:luaexpr()
    p:expect("]")
    p:expect("=")
    local value = p:expr()
    return ast.unspecialized.ExprCtorRecField {
      name_expr = name_expr,
      value = value,
      span = ast.span(start, p),
    }

  else
    local value = p:expr()
    return ast.unspecialized.ExprCtorListField {
      value = value,
      span = ast.span(start, p),
    }
  end
end

function parser.sep(p)
  return p:nextif(",") or p:nextif(";")
end

function parser.expr_ctor(p)
  local start = ast.save(p)
  local fields = terralib.newlist()
  p:expect("{")
  repeat
    if p:matches("}") then break end
    local field = p:field()
    fields:insert(field)
  until not p:sep()
  p:expect("}")

  return ast.unspecialized.ExprCtor {
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.fnargs(p)
  if p:nextif("(") then
    local args
    if not p:matches(")") then
      args = p:expr_list()
    else
      args = terralib.newlist()
    end
    p:expect(")")
    return args

  elseif p:matches("{") then
    local arg = p:expr_ctor()
    return terralib.newlist({arg})

  elseif p:matches(p.string) then
    local arg = p:expr_simple()
    return terralib.newlist({arg})

  else
    p:error("unexpected token in fnargs expression")
  end
end

function parser.expr_primary(p)
  local start = ast.save(p)
  local expr = p:expr_prefix()

  while true do
    if p:nextif(".") then
      local field_names = terralib.newlist()
      if p:nextif("partition") then
        field_names:insert("partition")
      elseif p:nextif("product") then
        field_names:insert("product")
      elseif p:nextif("{") then
        repeat
          if p:matches("}") then break end
          local field_name = p:next(p.name).value
          field_names:insert(field_name)
        until not p:sep()
        p:expect("}")
      else
        field_names:insert(p:next(p.name).value)
      end
      expr = ast.unspecialized.ExprFieldAccess {
        value = expr,
        field_names = field_names,
        span = ast.span(start, p),
      }

    elseif p:nextif("[") then
      local index = p:expr()
      p:expect("]")
      expr = ast.unspecialized.ExprIndexAccess {
        value = expr,
        index = index,
        span = ast.span(start, p),
      }

    elseif p:nextif(":") then
      local method_name = p:next(p.name).value
      local args = p:fnargs()
      expr = ast.unspecialized.ExprMethodCall {
        value = expr,
        method_name = method_name,
        args = args,
        span = ast.span(start, p),
      }

    elseif p:matches("(") or p:matches("{") or p:matches(p.string) then
      local args = p:fnargs()
      expr = ast.unspecialized.ExprCall {
        fn = expr,
        args = args,
        inline = "allow",
        span = ast.span(start, p),
      }

    else
      break
    end
  end

  return expr
end

function parser.expr_simple(p)
  local start = ast.save(p)
  if p:matches(p.number) then
    local token = p:next(p.number)
    return ast.unspecialized.ExprConstant {
      value = token.value,
      expr_type = token.valuetype,
      span = ast.span(start, p),
    }

  elseif p:matches(p.string) then
    local token = p:next(p.string)
    return ast.unspecialized.ExprConstant {
      value = token.value,
      expr_type = rawstring,
      span = ast.span(start, p),
    }

  elseif p:nextif("true") then
    return ast.unspecialized.ExprConstant {
      value = true,
      expr_type = bool,
      span = ast.span(start, p),
    }

  elseif p:nextif("false") then
    return ast.unspecialized.ExprConstant {
      value = false,
      expr_type = bool,
      span = ast.span(start, p),
    }

  elseif p:nextif("max") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprBinary {
      op = "max",
      lhs = lhs,
      rhs = rhs,
      span = ast.span(start, p),
    }

  elseif p:nextif("min") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprBinary {
      op = "min",
      lhs = lhs,
      rhs = rhs,
      span = ast.span(start, p),
    }

  elseif p:nextif("__context") then
    p:expect("(")
    p:expect(")")
    return ast.unspecialized.ExprRawContext {
      span = ast.span(start, p),
    }

  elseif p:nextif("__fields") then
    p:expect("(")
    local region = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprRawFields {
      region = region,
      span = ast.span(start, p),
    }

  elseif p:nextif("__physical") then
    p:expect("(")
    local region = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprRawPhysical {
      region = region,
      span = ast.span(start, p),
    }

  elseif p:nextif("__runtime") then
    p:expect("(")
    p:expect(")")
    return ast.unspecialized.ExprRawRuntime {
      span = ast.span(start, p),
    }

  elseif p:nextif("__raw") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprRawValue {
      value = value,
      span = ast.span(start, p),
    }

  elseif p:matches("__demand") or p:matches("__forbid") then
    local inline = false
    if p:nextif("__demand") then
      inline = "demand"
    elseif p:nextif("__forbid") then
      inline = "forbid"
    else
      assert(false)
    end

    p:expect("(")
    p:expect("__inline")
    p:expect(",")
    local call = p:expr()
    if not call:is(ast.unspecialized.ExprCall) then
      p:error("pragma '__demand(__inline, ...)' can only be used with calls")
    end
    call.inline = inline
    p:expect(")")
    return call

  elseif p:nextif("isnull") then
    p:expect("(")
    local pointer = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprIsnull {
      pointer = pointer,
      span = ast.span(start, p),
    }

  elseif p:nextif("new") then
    p:expect("(")
    local pointer_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.ExprNew {
      pointer_type_expr = pointer_type_expr,
      span = ast.span(start, p),
    }

  elseif p:nextif("null") then
    p:expect("(")
    local pointer_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.ExprNull {
      pointer_type_expr = pointer_type_expr,
      span = ast.span(start, p),
    }

  elseif p:nextif("dynamic_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprDynamicCast {
      type_expr = type_expr,
      value = value,
      span = ast.span(start, p),
    }

  elseif p:nextif("static_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprStaticCast {
      type_expr = type_expr,
      value = value,
      span = ast.span(start, p),
    }

  elseif p:nextif("ispace") then
    p:expect("(")
    local index_type_expr = p:luaexpr()
    p:expect(",")
    local extent = p:expr()
    local start_at = false
    if not p:matches(")") then
      p:expect(",")
      start_at = p:expr()
    end
    p:expect(")")
    return ast.unspecialized.ExprIspace {
      index_type_expr = index_type_expr,
      extent = extent,
      start = start_at,
      span = ast.span(start, p),
    }

  elseif p:nextif("region") then
    p:expect("(")
    local ispace = p:expr()
    p:expect(",")
    local fspace_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.ExprRegion {
      ispace = ispace,
      fspace_type_expr = fspace_type_expr,
      span = ast.span(start, p),
    }

  elseif p:nextif("partition") then
    p:expect("(")
    local disjointness_expr = p:luaexpr()
    p:expect(",")
    local region_type_expr = p:luaexpr()
    p:expect(",")
    local coloring = p:expr()
    p:expect(")")
    return ast.unspecialized.ExprPartition {
      disjointness_expr = disjointness_expr,
      region_type_expr = region_type_expr,
      coloring = coloring,
      span = ast.span(start, p),
    }

  elseif p:nextif("cross_product") then
    p:expect("(")
    local arg_type_exprs = terralib.newlist()
    repeat
      arg_type_exprs:insert(p:luaexpr())
    until not p:nextif(",")
    p:expect(")")
    return ast.unspecialized.ExprCrossProduct {
      arg_type_exprs = arg_type_exprs,
      span = ast.span(start, p),
    }

  elseif p:matches("{") then
    return p:expr_ctor()

  else
    return p:expr_primary()
  end
end

parser.expr_unary = function(precedence)
  return function(p)
    local start = ast.save(p)
    local op = p:next().type
    local rhs = p:expr(precedence)
    if op == "@" then
      return ast.unspecialized.ExprDeref {
        value = rhs,
        span = ast.span(start, p),
      }
    end
    return ast.unspecialized.ExprUnary {
      op = op,
      rhs = rhs,
      span = ast.span(start, p),
    }
  end
end

parser.expr_binary_left = function(p, lhs)
  local start = lhs.span.start
  local op = p:next().type
  local rhs = p:expr(op)
  return ast.unspecialized.ExprBinary {
    op = op,
    lhs = lhs,
    rhs = rhs,
    span = ast.span(start, p),
  }
end

parser.expr = parsing.Pratt()
  :prefix("@", parser.expr_unary(50))
  :prefix("-", parser.expr_unary(50))
  :prefix("not", parser.expr_unary(50))
  :infix("*", 40, parser.expr_binary_left)
  :infix("/", 40, parser.expr_binary_left)
  :infix("%", 40, parser.expr_binary_left)
  :infix("+", 30, parser.expr_binary_left)
  :infix("-", 30, parser.expr_binary_left)
  :infix("<", 20, parser.expr_binary_left)
  :infix(">", 20, parser.expr_binary_left)
  :infix("<=", 20, parser.expr_binary_left)
  :infix(">=", 20, parser.expr_binary_left)
  :infix("==", 20, parser.expr_binary_left)
  :infix("~=", 20, parser.expr_binary_left)
  :infix("and", 10, parser.expr_binary_left)
  :infix("or", 10, parser.expr_binary_left)
  :prefix(parsing.default, parser.expr_simple)

function parser.expr_lhs(p)
  local start = ast.save(p)
  if p:nextif("@") then
    local value = p:expr(50) -- Precedence for unary @
    return ast.unspecialized.ExprDeref {
      value = value,
      span = ast.span(start, p),
    }
  else
    return p:expr_primary()
  end
end

function parser.expr_list(p)
  local exprs = terralib.newlist()
  repeat
    exprs:insert(p:expr())
  until not p:nextif(",")
  return exprs
end

function parser.block(p)
  local start = ast.save(p)
  local block = terralib.newlist()
  while not (p:matches("end") or p:matches("elseif") or
             p:matches("else") or p:matches("until")) do
    local stat = p:stat()
    block:insert(stat)
    p:nextif(";")
  end
  return ast.unspecialized.Block {
    stats = block,
    span = ast.span(start, p),
  }
end

function parser.stat_if(p)
  local start = ast.save(p)
  p:expect("if")
  local cond = p:expr()
  p:expect("then")
  local then_block = p:block()
  local elseif_blocks = terralib.newlist()
  local elseif_start = ast.save(p)
  while p:nextif("elseif") do
    local elseif_cond = p:expr()
    p:expect("then")
    local elseif_block = p:block()
    elseif_blocks:insert(ast.unspecialized.StatElseif {
        cond = elseif_cond,
        block = elseif_block,
        span = ast.span(elseif_start, p),
    })
    elseif_start = ast.save(p)
  end
  local else_block = ast.unspecialized.Block {
    stats = terralib.newlist(),
    span = ast.empty_span(p),
  }
  if p:nextif("else") then
    else_block = p:block()
  end
  p:expect("end")
  return ast.unspecialized.StatIf {
    cond = cond,
    then_block = then_block,
    elseif_blocks = elseif_blocks,
    else_block = else_block,
    span = ast.span(start, p),
  }
end

function parser.stat_while(p)
  local start = ast.save(p)
  p:expect("while")
  local cond = p:expr()
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.StatWhile {
    cond = cond,
    block = block,
    span = ast.span(start, p),
  }
end

function parser.stat_for_num(p, start, name, type_expr)
  local values = p:expr_list()

  if #values < 2 or #values > 3 then
    p:error("for loop over numbers requires two or three values")
  end

  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.StatForNum {
    name = name,
    type_expr = type_expr,
    values = values,
    block = block,
    parallel = "allow",
    span = ast.span(start, p),
  }
end

function parser.stat_for_list(p, start, name, type_expr)
  local value = p:expr()

  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.StatForList {
    name = name,
    type_expr = type_expr,
    value = value,
    block = block,
    vectorize = "allow",
    span = ast.span(start, p),
  }
end

function parser.stat_for(p)
  local start = ast.save(p)
  p:expect("for")

  local name = p:expect(p.name).value
  local type_expr
  if p:nextif(":") then
    type_expr = p:luaexpr()
  else
    type_expr = function(env) end
  end

  if p:nextif("=") then
    return p:stat_for_num(start, name, type_expr)
  elseif p:nextif("in") then
    return p:stat_for_list(start, name, type_expr)
  else
    p:error("expected = or in")
  end
end

function parser.stat_repeat(p)
  local start = ast.save(p)
  p:expect("repeat")
  local block = p:block()
  p:expect("until")
  local until_cond = p:expr()
  return ast.unspecialized.StatRepeat {
    block = block,
    until_cond = until_cond,
    span = ast.span(start, p),
  }
end

function parser.stat_block(p)
  local start = ast.save(p)
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.StatBlock {
    block = block,
    span = ast.span(start, p),
  }
end

function parser.stat_var_unpack(p, start)
  p:expect("{")
  local names = terralib.newlist()
  local fields = terralib.newlist()
  repeat
    local name = p:next(p.name).value
    names:insert(name)
    if p:nextif("=") then
      fields:insert(p:next(p.name).value)
    else
      fields:insert(name)
    end
  until not p:nextif(",")
  p:expect("}")
  p:expect("=")
  local value = p:expr()
  return ast.unspecialized.StatVarUnpack {
    var_names = names,
    fields = fields,
    value = value,
    span = ast.span(start, p),
  }
end

function parser.stat_var(p)
  local start = ast.save(p)
  p:expect("var")
  if p:matches("{") then
    return p:stat_var_unpack(start)
  end

  local names = terralib.newlist()
  local type_exprs = terralib.newlist()
  repeat
    names:insert(p:next(p.name).value)
    if p:nextif(":") then
      type_exprs:insert(p:luaexpr())
    else
      type_exprs:insert(nil)
    end
  until not p:nextif(",")
  local values = terralib.newlist()
  if p:nextif("=") then
    values = p:expr_list()
  end
  return ast.unspecialized.StatVar {
    var_names = names,
    type_exprs = type_exprs,
    values = values,
    span = ast.span(start, p),
  }
end

function parser.stat_return(p)
  local start = ast.save(p)
  p:expect("return")
  local value = false
  if not (p:matches("end") or p:matches("elseif") or
          p:matches("else") or p:matches("until"))
  then
    value = p:expr()
  end
  return ast.unspecialized.StatReturn {
    value = value,
    span = ast.span(start, p),
  }
end

function parser.stat_break(p)
  local start = ast.save(p)
  p:expect("break")
  return ast.unspecialized.StatBreak {
    span = ast.span(start, p),
  }
end

parser.stat_expr_assignment = function(p, start, first_lhs)
  local lhs = terralib.newlist()
  lhs:insert(first_lhs)
  while p:nextif(",") do
    lhs:insert(p:expr_lhs())
  end

  local op
  -- Hack: Terra's lexer doesn't understand += as a single operator so
  -- for the moment read it as + followed by =.
  if p:lookahead("=") then
    if p:nextif("+") then
      op = "+"
    elseif p:nextif("-") then
      op = "-"
    elseif p:nextif("*") then
      op = "*"
    elseif p:nextif("/") then
      op = "/"
    elseif p:nextif("max") then
      op = "max"
    elseif p:nextif("min") then
      op = "min"
    else
      -- Fall through as if this were the normal = case.
    end
    p:expect("=")
  else
    p:expect("=")
  end

  local rhs = p:expr_list()
  if op then
    return ast.unspecialized.StatReduce {
      op = op,
      lhs = lhs,
      rhs = rhs,
      span = ast.span(start, p),
    }
  else
    return ast.unspecialized.StatAssignment {
      lhs = lhs,
      rhs = rhs,
      span = ast.span(start, p),
    }
  end
end

function parser.stat_expr(p)
  local start = ast.save(p)
  local first_lhs = p:expr_lhs()
  if p:matches(",") or
    p:matches("=") or
    (p:matches("+") and p:lookahead("=")) or
    (p:matches("-") and p:lookahead("=")) or
    (p:matches("*") and p:lookahead("=")) or
    (p:matches("/") and p:lookahead("=")) or
    (p:matches("max") and p:lookahead("=")) or
    (p:matches("min") and p:lookahead("="))
  then
    return p:stat_expr_assignment(start, first_lhs)
  else
    return ast.unspecialized.StatExpr {
      expr = first_lhs,
      span = ast.span(start, p),
    }
  end
end

function parser.stat_pragma(p)
  local start = ast.save(p)
  local parallel = "allow"
  local vectorize = "allow"
  local inline = "allow"

  local pragma = false
  if p:nextif("__demand") then
    pragma = "demand"
  elseif p:nextif("__forbid") then
    pragma = "forbid"
  end

  p:expect("(")
  if p:nextif("__parallel") then
    parallel = pragma
  elseif p:nextif("__vectorize") then
    vectorize = pragma
  elseif p:nextif("__inline") then
    inline = pragma
  else
    p:error("expected __parallel, __vectorize, or __inline")
  end

  if p:matches(",") then
    p:expect(",")
    local call = p:expr()
    p:expect(")")
    if not call:is(ast.unspecialized.ExprCall) then
      p:error("pragma can only be used with call expressions")
    end
    call.inline = inline
    return ast.unspecialized.StatExpr {
      expr = call,
      span = call.span,
    }
  else
    p:expect(")")
    local for_stat = p:stat_for()
    for_stat.parallel = parallel
    for_stat.vectorize = vectorize
    return for_stat
  end
end

function parser.stat(p)
  if p:matches("if") then
    return p:stat_if()

  elseif p:matches("while") then
    return p:stat_while()

  elseif p:matches("__demand") or p:matches("__forbid") then
    return p:stat_pragma()

  elseif p:matches("for") then
    return p:stat_for()

  elseif p:matches("repeat") then
    return p:stat_repeat()

  elseif p:matches("do") then
    return p:stat_block()

  elseif p:matches("var") then
    return p:stat_var()

  elseif p:matches("return") then
    return p:stat_return()

  elseif p:matches("break") then
    return p:stat_break()

  else
    return p:stat_expr()
  end
end

function parser.stat_task_params(p)
  p:expect("(")
  local params = terralib.newlist()
  if not p:matches(")") then
    repeat
      local start = ast.save(p)
      local param_name = p:expect(p.name).value
      p:expect(":")
      local param_type = p:luaexpr()
      params:insert(ast.unspecialized.StatTaskParam {
          param_name = param_name,
          type_expr = param_type,
          span = ast.span(start, p),
      })
    until not p:nextif(",")
  end
  p:expect(")")
  return params
end

function parser.stat_task_return(p)
  if p:nextif(":") then
    return p:luaexpr()
  end
  return function(env) return std.untyped end
end

function parser.privilege_region_field(p)
  local start = ast.save(p)
  local field_name = p:expect(p.name).value
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:privilege_region_fields()
  end
  return ast.unspecialized.PrivilegeRegionField {
    field_name = field_name,
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.privilege_region_fields(p)
  local fields = terralib.newlist()
  if p:nextif("{") then
    repeat
      if p:matches("}") then break end
      fields:insert(p:privilege_region_field())
    until not p:sep()
    p:expect("}")
  else
    fields:insert(p:privilege_region_field())
  end
  return fields
end

function parser.privilege_region(p)
  local start = ast.save(p)
  local region_name = p:expect(p.name).value
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:privilege_region_fields()
  end
  return ast.unspecialized.PrivilegeRegion {
    region_name = region_name,
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.privilege(p)
  local start = ast.save(p)
  local privilege
  local op = false
  if p:nextif("reads") then
    privilege = "reads"
  elseif p:nextif("writes") then
    privilege = "writes"
  elseif p:nextif("reduces") then
    privilege = "reduces"
    if p:nextif("+") then
      op = "+"
    elseif p:nextif("-") then
      op = "-"
    elseif p:nextif("*") then
      op = "*"
    elseif p:nextif("/") then
      op = "/"
    elseif p:nextif("max") then
      op = "max"
    elseif p:nextif("min") then
      op = "min"
    else
      p:error("expected operator")
    end
  else
    p:error("expected reads or writes")
  end

  p:expect("(")
  local regions = terralib.newlist()
  repeat
    local region = p:privilege_region()
    regions:insert(region)
  until not p:nextif(",")
  p:expect(")")

  return ast.unspecialized.Privilege {
    privilege = privilege,
    op = op,
    regions = regions,
    span = ast.span(start, p),
  }
end

function parser.constraint(p)
  local start = ast.save(p)
  local lhs = p:expect(p.name).value
  local op
  if p:nextif("<=") then
    op = "<="
  elseif p:nextif("*") then
    op = "*"
  else
    p:error("unexpected token in constraint")
  end
  local rhs = p:expect(p.name).value
  return ast.unspecialized.Constraint {
    lhs = lhs,
    op = op,
    rhs = rhs,
    span = ast.span(start, p),
  }
end

function parser.stat_task_privileges_and_constraints(p)
  local privileges = terralib.newlist()
  local constraints = terralib.newlist()
  if p:nextif("where") then
    repeat
      if p:matches("reads") or p:matches("writes") or p:matches("reduces") then
        privileges:insert(p:privilege())
      else
        constraints:insert(p:constraint())
      end
    until not p:nextif(",")
    p:expect("do")
  end
  return privileges, constraints
end

function parser.stat_task(p, task_opts)
  local start = ast.save(p)
  p:expect("task")
  local name = p:expect(p.name).value
  local params = p:stat_task_params()
  local return_type = p:stat_task_return()
  local privileges, constraints = p:stat_task_privileges_and_constraints()
  local body = p:block()
  p:expect("end")

  return ast.unspecialized.StatTask {
    name = name,
    params = params,
    return_type_expr = return_type,
    privileges = privileges,
    constraints = constraints,
    body = body,
    inline = task_opts.inline,
    cuda = task_opts.cuda,
    span = ast.span(start, p),
  }
end

function parser.stat_fspace_params(p)
  local params = terralib.newlist()
  if p:nextif("(") then
    if not p:matches(")") then
      repeat
        local start = ast.save(p)
        local param_name = p:expect(p.name).value
        p:expect(":")
        local param_type = p:luaexpr()
        params:insert(ast.unspecialized.StatFspaceParam {
          param_name = param_name,
          type_expr = param_type,
          span = ast.span(start, p),
        })
      until not p:nextif(",")
    end
    p:expect(")")
  end
  return params
end

function parser.stat_fspace_fields(p)
  local fields = terralib.newlist()
  p:expect("{")
  repeat
    if p:matches("}") then break end

    local start = ast.save(p)
    local field_name = p:expect(p.name).value
    p:expect(":")
    local field_type = p:luaexpr()
    fields:insert(ast.unspecialized.StatFspaceField {
      field_name = field_name,
      type_expr = field_type,
      span = ast.span(start, p),
    })
  until not p:sep()
  p:expect("}")
  return fields
end

function parser.stat_fspace_constraints(p)
  local constraints = terralib.newlist()
  if p:nextif("where") then
    repeat
      constraints:insert(p:constraint())
    until not p:nextif(",")
    p:expect("end")
  end
  return constraints
end

function parser.stat_fspace(p)
  local start = ast.save(p)
  p:expect("fspace")
  local name = p:expect(p.name).value
  local params = p:stat_fspace_params()
  local fields = p:stat_fspace_fields()
  local constraints = p:stat_fspace_constraints()

  return ast.unspecialized.StatFspace {
    name = name,
    params = params,
    fields = fields,
    constraints = constraints,
    span = ast.span(start, p),
  }
end

function parser.stat_top(p)
  if p:matches("task") then
    return p:stat_task { inline = false, cuda = false }

  elseif p:matches("fspace") then
    return p:stat_fspace()

  elseif p:matches("__demand") then
    local task_opts = { inline = false, cuda = false }
    p:expect("__demand")
    p:expect("(")
    repeat
      if p:matches("__inline") then
        p:expect("__inline")
        task_opts.inline = true
      elseif p:matches("__cuda") then
        p:expect("__cuda")
        if not terralib.cudacompile then
          p:error("CUDA tasks are demanded, but CUDA is not enabled")
        end
        task_opts.cuda = { unrolling_factor = 1 }
      elseif p:matches("__unroll") then
        if not task_opts.cuda then
          p:error("unexpected unrolling factor for CPU task")
        end
        p:expect("__unroll")
        p:expect("(")
        local token = p:next(p.number)
        p:expect(")")
        if token.value ~= math.floor(token.value) then
          p:error("unrolling factor should be an integer")
        end
        task_opts.cuda.unrolling_factor = token.value
      end
    until not p:nextif(",")
    p:expect(")")
    return p:stat_task(task_opts)

  else
    p:error("unexpected token in top-level statement")
  end
end

function parser:parse(lex)
  return parsing.Parse(self, lex, "stat_top")
end

return parser
