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

function parser.option_level(p)
  if p:nextif("__allow") then
    return ast.options.Allow { value = false }
  elseif p:nextif("__demand") then
    return ast.options.Demand { value = false }
  elseif p:nextif("__forbid") then
    return ast.options.Forbid { value = false }
  else
    return false
  end
end

function parser.option_values(p)
  if p:nextif("(") then
    p:expect("__unroll")
    p:expect("(")
    local value = p:next(p.number)
    p:expect(")")
    if value.value ~= math.floor(value.value) then
      p:error("unroll factor should be an integer")
    end
    p:expect(")")
    p:expect(")")
    return ast.options.Unroll { value = value }
  else
    return false
  end
end

function parser.option_name(p)
  if p:nextif("__cuda") then
    local values = p:option_values()
    return "cuda", values
  elseif p:nextif("__inline") then
    return "inline"
  elseif p:nextif("__parallel") then
    return "parallel"
  elseif p:nextif("__spmd") then
    return "spmd"
  elseif p:nextif("__vectorize") then
    return "vectorize"
  end
end

function parser.options(p, allow_expr, allow_stat)
  assert(allow_expr or allow_stat)
  local options = ast.default_options()

  local level = p:option_level()
  if not level then return options end

  p:expect("(")
  local name = p:option_name()
  options = options { [name] = level }

  if allow_expr and p:nextif(",") then
    local expr = p:expr()
    p:expect(")")
    return expr { options = options }
  elseif allow_stat then
    p:expect(")")
    return options
  else
    -- throw an error if this was supposed to be an expression.
    p:expect(",")
  end
end

function parser.expr_prefix(p)
  local start = ast.save(p)
  if p:nextif("(") then
    local expr = p:expr()
    p:expect(")")
    return expr

  elseif p:nextif("[") then
    local expr = p:luaexpr()
    p:expect("]")
    return ast.unspecialized.expr.Escape {
      expr = expr,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:matches(p.name) then
    local name = p:next(p.name).value
    p:ref(name)
    return ast.unspecialized.expr.ID {
      name = name,
      options = ast.default_options(),
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
    return ast.unspecialized.expr.CtorRecField {
      name_expr = function(env) return name end,
      value = value,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("[") then
    local name_expr = p:luaexpr()
    p:expect("]")
    p:expect("=")
    local value = p:expr()
    return ast.unspecialized.expr.CtorRecField {
      name_expr = name_expr,
      value = value,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  else
    local value = p:expr()
    return ast.unspecialized.expr.CtorListField {
      value = value,
      options = ast.default_options(),
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

  return ast.unspecialized.expr.Ctor {
    fields = fields,
    options = ast.default_options(),
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
      expr = ast.unspecialized.expr.FieldAccess {
        value = expr,
        field_names = field_names,
        options = ast.default_options(),
        span = ast.span(start, p),
      }

    elseif p:nextif("[") then
      local index = p:expr()
      p:expect("]")
      expr = ast.unspecialized.expr.IndexAccess {
        value = expr,
        index = index,
        options = ast.default_options(),
        span = ast.span(start, p),
      }

    elseif p:nextif(":") then
      local method_name = p:next(p.name).value
      local args = p:fnargs()
      expr = ast.unspecialized.expr.MethodCall {
        value = expr,
        method_name = method_name,
        args = args,
        options = ast.default_options(),
        span = ast.span(start, p),
      }

    elseif p:matches("(") or p:matches("{") or p:matches(p.string) then
      local args = p:fnargs()
      expr = ast.unspecialized.expr.Call {
        fn = expr,
        args = args,
        options = ast.default_options(),
        span = ast.span(start, p),
      }

    else
      break
    end
  end

  return expr
end

function parser.expr_simple(p)
  local options = p:options(true, false)
  if options:is(ast.unspecialized.expr) then
    return options
  end

  local start = ast.save(p)
  if p:matches(p.number) then
    local token = p:next(p.number)
    return ast.unspecialized.expr.Constant {
      value = token.value,
      expr_type = token.valuetype,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:matches(p.string) then
    local token = p:next(p.string)
    return ast.unspecialized.expr.Constant {
      value = token.value,
      expr_type = rawstring,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("true") then
    return ast.unspecialized.expr.Constant {
      value = true,
      expr_type = bool,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("false") then
    return ast.unspecialized.expr.Constant {
      value = false,
      expr_type = bool,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("max") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Binary {
      op = "max",
      lhs = lhs,
      rhs = rhs,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("min") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Binary {
      op = "min",
      lhs = lhs,
      rhs = rhs,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__context") then
    p:expect("(")
    p:expect(")")
    return ast.unspecialized.expr.RawContext {
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__fields") then
    p:expect("(")
    local region = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawFields {
      region = region,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__physical") then
    p:expect("(")
    local region = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawPhysical {
      region = region,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__runtime") then
    p:expect("(")
    p:expect(")")
    return ast.unspecialized.expr.RawRuntime {
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__raw") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawValue {
      value = value,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("isnull") then
    p:expect("(")
    local pointer = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Isnull {
      pointer = pointer,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("new") then
    p:expect("(")
    local pointer_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.expr.New {
      pointer_type_expr = pointer_type_expr,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("null") then
    p:expect("(")
    local pointer_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.expr.Null {
      pointer_type_expr = pointer_type_expr,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("dynamic_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.DynamicCast {
      type_expr = type_expr,
      value = value,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("static_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.StaticCast {
      type_expr = type_expr,
      value = value,
      options = ast.default_options(),
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
    return ast.unspecialized.expr.Ispace {
      index_type_expr = index_type_expr,
      extent = extent,
      start = start_at,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("region") then
    p:expect("(")
    local ispace = p:expr()
    p:expect(",")
    local fspace_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.expr.Region {
      ispace = ispace,
      fspace_type_expr = fspace_type_expr,
      options = ast.default_options(),
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
    return ast.unspecialized.expr.Partition {
      disjointness_expr = disjointness_expr,
      region_type_expr = region_type_expr,
      coloring = coloring,
      options = ast.default_options(),
      span = ast.span(start, p),
    }

  elseif p:nextif("cross_product") then
    p:expect("(")
    local arg_type_exprs = terralib.newlist()
    repeat
      arg_type_exprs:insert(p:luaexpr())
    until not p:nextif(",")
    p:expect(")")
    return ast.unspecialized.expr.CrossProduct {
      arg_type_exprs = arg_type_exprs,
      options = ast.default_options(),
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
      return ast.unspecialized.expr.Deref {
        value = rhs,
        options = ast.default_options(),
        span = ast.span(start, p),
      }
    end
    return ast.unspecialized.expr.Unary {
      op = op,
      rhs = rhs,
      options = ast.default_options(),
      span = ast.span(start, p),
    }
  end
end

parser.expr_binary_left = function(p, lhs)
  local start = lhs.span.start
  local op = p:next().type
  local rhs = p:expr(op)
  return ast.unspecialized.expr.Binary {
    op = op,
    lhs = lhs,
    rhs = rhs,
    options = ast.default_options(),
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
    return ast.unspecialized.expr.Deref {
      value = value,
      options = ast.default_options(),
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

function parser.stat_if(p, options)
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
    elseif_blocks:insert(ast.unspecialized.stat.Elseif {
        cond = elseif_cond,
        block = elseif_block,
        options = options,
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
  return ast.unspecialized.stat.If {
    cond = cond,
    then_block = then_block,
    elseif_blocks = elseif_blocks,
    else_block = else_block,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_while(p, options)
  local start = ast.save(p)
  p:expect("while")
  local cond = p:expr()
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.While {
    cond = cond,
    block = block,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_for_num(p, start, name, type_expr, options)
  local values = p:expr_list()

  if #values < 2 or #values > 3 then
    p:error("for loop over numbers requires two or three values")
  end

  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.ForNum {
    name = name,
    type_expr = type_expr,
    values = values,
    block = block,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_for_list(p, start, name, type_expr, options)
  local value = p:expr()

  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.ForList {
    name = name,
    type_expr = type_expr,
    value = value,
    block = block,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_for(p, options)
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
    return p:stat_for_num(start, name, type_expr, options)
  elseif p:nextif("in") then
    return p:stat_for_list(start, name, type_expr, options)
  else
    p:error("expected = or in")
  end
end

function parser.stat_repeat(p, options)
  local start = ast.save(p)
  p:expect("repeat")
  local block = p:block()
  p:expect("until")
  local until_cond = p:expr()
  return ast.unspecialized.stat.Repeat {
    block = block,
    until_cond = until_cond,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_must_epoch(p, options)
  local start = ast.save(p)
  p:expect("must_epoch")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.MustEpoch {
    block = block,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_block(p, options)
  local start = ast.save(p)
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.Block {
    block = block,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_var_unpack(p, start, options)
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
  return ast.unspecialized.stat.VarUnpack {
    var_names = names,
    fields = fields,
    value = value,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_var(p, options)
  local start = ast.save(p)
  p:expect("var")
  if p:matches("{") then
    return p:stat_var_unpack(start, options)
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
  return ast.unspecialized.stat.Var {
    var_names = names,
    type_exprs = type_exprs,
    values = values,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_return(p, options)
  local start = ast.save(p)
  p:expect("return")
  local value = false
  if not (p:matches("end") or p:matches("elseif") or
          p:matches("else") or p:matches("until"))
  then
    value = p:expr()
  end
  return ast.unspecialized.stat.Return {
    value = value,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_break(p, options)
  local start = ast.save(p)
  p:expect("break")
  return ast.unspecialized.stat.Break {
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_expr_assignment(p, start, first_lhs, options)
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
    return ast.unspecialized.stat.Reduce {
      op = op,
      lhs = lhs,
      rhs = rhs,
      options = options,
      span = ast.span(start, p),
    }
  else
    return ast.unspecialized.stat.Assignment {
      lhs = lhs,
      rhs = rhs,
      options = options,
      span = ast.span(start, p),
    }
  end
end

function parser.stat_expr(p, options)
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
    return p:stat_expr_assignment(start, first_lhs, options)
  else
    return ast.unspecialized.stat.Expr {
      expr = first_lhs,
      options = options,
      span = ast.span(start, p),
    }
  end
end

function parser.stat(p)
  local options = p:options(true, true)
  if options:is(ast.unspecialized.expr) then
    return ast.unspecialized.stat.Expr {
      expr = options,
      options = ast.default_options(),
      span = options.span,
    }
  end

  if p:matches("if") then
    return p:stat_if(options)

  elseif p:matches("while") then
    return p:stat_while(options)

  elseif p:matches("for") then
    return p:stat_for(options)

  elseif p:matches("repeat") then
    return p:stat_repeat(options)

  elseif p:matches("must_epoch") then
    return p:stat_must_epoch(options)

  elseif p:matches("do") then
    return p:stat_block(options)

  elseif p:matches("var") then
    return p:stat_var(options)

  elseif p:matches("return") then
    return p:stat_return(options)

  elseif p:matches("break") then
    return p:stat_break(options)

  else
    return p:stat_expr(options)
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
      params:insert(ast.unspecialized.stat.TaskParam {
          param_name = param_name,
          type_expr = param_type,
          options = ast.default_options(),
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

function parser.region_field(p)
  local start = ast.save(p)
  local field_name = p:expect(p.name).value
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:region_fields()
  end
  return ast.unspecialized.region.Field {
    field_name = field_name,
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.region_fields(p)
  local fields = terralib.newlist()
  if p:nextif("{") then
    repeat
      if p:matches("}") then break end
      fields:insert(p:region_field())
    until not p:sep()
    p:expect("}")
  else
    fields:insert(p:region_field())
  end
  return fields
end

function parser.region_root(p)
  local start = ast.save(p)
  local region_name = p:expect(p.name).value
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:region_fields()
  end
  return ast.unspecialized.region.Root {
    region_name = region_name,
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.regions(p)
  local regions = terralib.newlist()
  repeat
    local region = p:region_root()
    regions:insert(region)
  until not p:nextif(",")
  return regions
end

function parser.region_bare(p)
  local start = ast.save(p)
  local region_name = p:expect(p.name).value
  return ast.unspecialized.region.Bare {
    region_name = region_name,
    span = ast.span(start, p),
  }
end

function parser.is_privilege_kind(p)
  return p:matches("reads") or p:matches("writes") or p:matches("reduces")
end

function parser.privilege_kind(p)
  local start = ast.save(p)
  if p:nextif("reads") then
    return ast.unspecialized.privilege_kind.Reads { span = ast.span(start, p) }
  elseif p:nextif("writes") then
    return ast.unspecialized.privilege_kind.Writes { span = ast.span(start, p) }
  elseif p:nextif("reduces") then
    local op = false
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
    return ast.unspecialized.privilege_kind.Reduces {
      op = op,
      span = ast.span(start, p),
    }
  else
    p:error("expected privilege")
  end
end

function parser.is_coherence_kind(p)
  return p:matches("exclusive") or p:matches("atomic") or
    p:matches("simultaneous") or p:matches("relaxed")
end

function parser.coherence_kind(p)
  local start = ast.save(p)
  if p:nextif("exclusive") then
    return ast.unspecialized.coherence_kind.Exclusive {
      span = ast.span(start, p),
    }
  elseif p:nextif("atomic") then
    return ast.unspecialized.coherence_kind.Atomic {
      span = ast.span(start, p),
    }
  elseif p:nextif("simultaneous") then
    return ast.unspecialized.coherence_kind.Simultaneous {
      span = ast.span(start, p),
    }
  elseif p:nextif("relaxed") then
    return ast.unspecialized.coherence_kind.Relaxed {
      span = ast.span(start, p),
    }
  else
    p:error("expected coherence mode")
  end
end

function parser.privilege_and_coherence_kinds(p)
  local privileges = terralib.newlist()
  local coherence_modes = terralib.newlist()
  while not p:matches("(") do
    if p:is_privilege_kind() then
      privileges:insert(p:privilege_kind())
    elseif p:is_coherence_kind() then
      coherence_modes:insert(p:coherence_kind())
    else
      p:error("expected privilege or coherence mode")
    end
  end
  return privileges, coherence_modes
end

function parser.privilege_and_coherence(p)
  local start = ast.save(p)
  local privileges, coherence_modes = p:privilege_and_coherence_kinds()
  p:expect("(")
  local regions = p:regions()
  p:expect(")")

  local privilege = ast.unspecialized.Privilege {
    privileges = privileges,
    regions = regions,
    span = ast.span(start, p),
  }
  local coherence = ast.unspecialized.Coherence {
    coherence_modes = coherence_modes,
    regions = regions,
    span = ast.span(start, p),
  }
  return privilege, coherence
end

function parser.constraint_kind(p)
  local start = ast.save(p)
  if p:nextif("<=") then
    return ast.unspecialized.constraint_kind.Subregion {
      span = ast.span(start, p),
    }
  elseif p:nextif("*") then
    return ast.unspecialized.constraint_kind.Disjointness {
      span = ast.span(start, p),
    }
  else
    p:error("unexpected token in constraint")
  end
end

function parser.constraint(p)
  local start = ast.save(p)
  local lhs = p:region_bare()
  local op = p:constraint_kind()
  local rhs = p:region_bare()
  return ast.unspecialized.Constraint {
    lhs = lhs,
    op = op,
    rhs = rhs,
    span = ast.span(start, p),
  }
end

function parser.stat_task_privileges_and_constraints(p)
  local privileges = terralib.newlist()
  local coherence_modes = terralib.newlist()
  local constraints = terralib.newlist()
  if p:nextif("where") then
    repeat
      if p:is_privilege_kind() or p:is_coherence_kind() then
        local privilege, coherence = p:privilege_and_coherence()
        privileges:insert(privilege)
        coherence_modes:insert(coherence)
      else
        constraints:insert(p:constraint())
      end
    until not p:nextif(",")
    p:expect("do")
  end
  return privileges, coherence_modes, constraints
end

function parser.stat_task(p, options)
  local start = ast.save(p)
  p:expect("task")
  local name = p:expect(p.name).value
  local params = p:stat_task_params()
  local return_type = p:stat_task_return()
  local privileges, coherence_modes, constraints =
    p:stat_task_privileges_and_constraints()
  local body = p:block()
  p:expect("end")

  return ast.unspecialized.stat.Task {
    name = name,
    params = params,
    return_type_expr = return_type,
    privileges = privileges,
    coherence_modes = coherence_modes,
    constraints = constraints,
    body = body,
    options = options,
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
        params:insert(ast.unspecialized.stat.FspaceParam {
          param_name = param_name,
          type_expr = param_type,
          options = ast.default_options(),
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
    fields:insert(ast.unspecialized.stat.FspaceField {
      field_name = field_name,
      type_expr = field_type,
      options = ast.default_options(),
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

function parser.stat_fspace(p, options)
  local start = ast.save(p)
  p:expect("fspace")
  local name = p:expect(p.name).value
  local params = p:stat_fspace_params()
  local fields = p:stat_fspace_fields()
  local constraints = p:stat_fspace_constraints()

  return ast.unspecialized.stat.Fspace {
    name = name,
    params = params,
    fields = fields,
    constraints = constraints,
    options = options,
    span = ast.span(start, p),
  }
end

function parser.stat_top(p)
  local options = p:options(false, true)

  local stat
  if p:matches("task") then
    return p:stat_task(options)

  elseif p:matches("fspace") then
    return p:stat_fspace(options)

  else
    p:error("unexpected token in top-level statement")
  end
end

function parser:parse(lex)
  return parsing.Parse(self, lex, "stat_top")
end

return parser
