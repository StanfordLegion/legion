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

-- Bishop Parser

local parsing = require("parsing")
local ast = require("bishop/ast")

local parser = {}

function parser.value(p)
  local value
  if p:matches(p.number) then
    local token = p:next(p.number)
    value = ast.unspecialized.expr.Constant {
      constant = token.value,
      type = token.valuetype,
    }
  elseif p:nextif("$") then
    local token = p:next(p.name)
    value = ast.unspecialized.expr.Variable {
      id = "$" .. token.value,
    }
  elseif p:matches(p.name) then
    local token = p:next(p.name)
    value = ast.unspecialized.expr.Constant {
      constant = token.value,
      type = "keyword",
    }
  else
    p:error("unexpected type of value")
  end
  return value
end

function parser.expr(p)
  local expr = p:value()
  while true do
    if p:nextif("[") then
      if p:matches(p.name) and p:lookahead("=") then
        local constraints = p:constraints()
        expr = ast.unspecialized.expr.Index {
          value = expr,
          index = constraints,
        }
      else
        local value = p:value()
        expr = ast.unspecialized.expr.Index {
          value = expr,
          index = value,
        }
      end
      p:expect("]")
    elseif p:nextif(".") then
      local token = p:next(p.name)
      expr = ast.unspecialized.expr.Field {
        value = expr,
        field = token.value,
      }
    else
      break
    end
  end
  return expr
end

function parser.property(p)
  local field = p:next(p.name)
  p:expect(":")
  local value = p:expr()
  p:expect(";")
  return ast.unspecialized.Property {
    field = field.value,
    value = value,
  }
end

function parser.properties(p)
  local properties = terralib.newlist()
  while not p:matches("}") do
    properties:insert(p:property())
  end
  return properties
end

function parser.constraint(p)
  local field = p:next(p.name)
  p:expect("=")
  local value = p:expr()
  return ast.unspecialized.Constraint {
    field = field.value,
    value = value,
  }
end

function parser.constraints(p)
  local constraints = terralib.newlist()
  if not p:matches("]") then
    constraints:insert(p:constraint())
    while p:matches("and") do
      p:expect("and")
      constraints:insert(p:constraint())
    end
  end
  return constraints
end

function parser.element(p)
  local element
  if p:nextif("task") then
    element = "task"
  elseif p:nextif("region") then
    element = "region"
  elseif p:nextif("for") then
    element = "for"
  elseif p:nextif("while") then
    element = "while"
  elseif p:nextif("do") then
    element = "do"
  else
    p:error("unexpected element name")
  end

  local name = terralib.newlist()
  local classes = terralib.newlist()
  local constraints = terralib.newlist()

  while p:matches("#") or p:matches(".") or p:matches("[") do
    if p:nextif("#") then
      if #name > 0 then
        p:error("selector cannot have multiple names")
      end
      name:insert(p:next(p.name).value)
    elseif p:nextif(".") then
      classes:insert(p:next(p.name).value)
    elseif p:nextif("[") then
      constraints:insertall(p:constraints())
      p:expect("]")
    else
      p:error("unexpected token")
    end
  end

  return ast.unspecialized.Element {
    type = element,
    name = name,
    classes = classes,
    constraints = constraints,
  }
end

function parser.selector(p)
  local elements = terralib.newlist()
  repeat
    elements:insert(p:element())
  until p:matches(",") or p:matches("{")
  return ast.unspecialized.Selector {
    elements = elements,
  }
end

function parser.selectors(p)
  local selectors = terralib.newlist()
  -- every rule should have at least one selector
  selectors:insert(p:selector())
  while p:matches(",") do
    p:expect(",")
    selectors:insert(p:selector())
    if p:matches("{") then break end
  end
  return selectors
end

function parser.rule(p)
  local selectors = p:selectors()
  p:expect("{")
  local properties = p:properties()
  p:expect("}")
  return ast.unspecialized.Rule {
    selectors = selectors,
    properties = properties,
  }
end

function parser.rules(p)
  local rules = terralib.newlist()
  while not p:matches("end") do
    rules:insert(p:rule())
  end
  return rules
end

function parser.top(p)
  if not p:matches("bishop") then
    p:error("unexpected token in top-level statement")
  end
  p:expect("bishop")

  local rules = p:rules()

  if not p:matches("end") then
    p:error("unexpected token in top-level statement")
  end
  p:expect("end")

  return ast.unspecialized.Rules {
    rules = rules,
  }
end

function parser:parse(lex)
  return parsing.Parse(self, lex, "top")
end

return parser
