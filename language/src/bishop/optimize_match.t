-- Copyright 2016 Stanford University, NVIDIA Corporation
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


-- Bishop Compilation Pass for Optimizing Rule Matching

local ast = require("bishop/ast")
local log = require("bishop/log")
local std = require("bishop/std")
local data = require("regent/data")
local automata = require("bishop/automata")
local regex = require("bishop/regex")

local task_id_map = {}
local function fetch_task_ids()
  local regent_std = require("regent/std")
  for k, v in pairs(_G) do
    if regent_std.is_task(v) then
      task_id_map[k] = v.taskid:asvalue()
    end
  end
end

local function traverse_element(rules, fn)
  local results = terralib.newlist()
  for ridx = 1, #rules do
    local rule = rules[ridx]
    assert(#rule.selectors == 1)
    local elements = terralib.newlist()
    local num_elements = #rule.selectors[1].elements
    if rule.rule_type == "region" then num_elements = num_elements - 1 end
    for eidx = 1, num_elements do
      local r = fn(rule.selectors[1].elements[eidx])
      if r then elements:insert(r) end
    end
    if #elements > 0 then results:insert(elements) end
  end
  if #results > 0 then return results end
end

local function collect_symbols(rules)
  -- collect all symbol hashes to assign consistent numbers
  local num_tasks = 0
  local all_task_ids = {}
  local num_constraints = 0
  local all_constraint_ids = {}
  local num_classes = 0
  local all_class_ids = {}

  for _, task_id in pairs(task_id_map) do
    all_task_ids[task_id] = task_id
    num_tasks = num_tasks + 1
  end
  traverse_element(rules, function(elem)
    elem.constraints:map(function(constraint)
      all_constraint_ids[constraint:unparse()] = num_constraints
      num_constraints = num_constraints + 1
    end)
    elem.classes:map(function(class)
      all_class_ids[class] = num_classes
      num_classes = num_classes + 1
    end)
  end)

  local all_ids = {}
  for hash, id in pairs(all_task_ids) do
    all_ids[hash] = id
  end
  for hash, id in pairs(all_constraint_ids) do
    all_ids[hash] = id + num_tasks
  end
  for hash, id in pairs(all_class_ids) do
    all_ids[hash] = id + num_tasks + num_constraints
  end

  local all_symbols = {}
  local all_task_symbols = {}
  local function update_symbol(hash, symbol)
    local id = all_ids[hash]
    if not all_symbols[id] then
      symbol.value = id
      all_symbols[id] = symbol
    end
    return id
  end

  local selectors = traverse_element(rules, function(elem)
    local name = elem.name:map(function(name)
      local id = update_symbol(task_id_map[name],
        regex.symbol.TaskId {
          value = 0,
          task_name = name,
        })
      all_task_symbols[id] = all_symbols[id]
      return id
    end)
    local classes = elem.classes:map(function(class)
      return update_symbol(class,
        regex.symbol.Class {
          value = 0,
          class_name = class,
        })
    end)
    local constraints = elem.constraints:map(function(constraint)
      return update_symbol(constraint:unparse(),
        regex.symbol.Constraint {
          value = 0,
          constraint = constraint,
        })
    end)
    return {
      name = name,
      classes = classes,
      constraints = constraints,
    }
  end)

  -- collect symbols per task
  local symbols_by_task = {}
  for _, task_symbol in pairs(all_task_symbols) do
    symbols_by_task[task_symbol] = {}
  end
  selectors:map(function(elements)
    for idx = 1, #elements do
      local elem = elements[idx]
      local task_symbols = all_task_symbols
      if #elem.name == 1 then
        task_symbols = {}
        task_symbols[elem.name[1]] = all_task_symbols[elem.name[1]]
      end
      for _, task_symbol in pairs(task_symbols) do
        elem.classes:map(function(id)
          local symbol = all_symbols[id]
          symbols_by_task[task_symbol][id] = all_symbols[symbol]
        end)
        elem.constraints:map(function(id)
          symbols_by_task[task_symbol][id] = all_symbols[id]
        end)
      end
    end
  end)

  return selectors, all_symbols, all_task_symbols, symbols_by_task
end

local function translate_to_regex(selectors, all_symbols, all_task_symbols, symbols_by_task)
  local prefix_by_task = terralib.newlist()
  for _, task_symbol in pairs(all_task_symbols) do
    local values = terralib.newlist()
    values:insert(task_symbol)
    for _, symbol in pairs(symbols_by_task[task_symbol]) do
      values:insert(regex.expr.Kleene { value = symbol })
    end
    prefix_by_task:insert(regex.expr.Concat { values = values })
  end

  local prefix = regex.expr.Kleene {
    value = regex.expr.Disj {
      values = prefix_by_task,
    },
  }

  return selectors:map(function(elements)
    local values = terralib.newlist()
    values:insert(prefix)
    for idx = 1, #elements do
      local elem = elements[idx]
      local elem_exprs = terralib.newlist()

      local task_symbols = all_task_symbols
      if #elem.name == 1 then
        task_symbols = {}
        task_symbols[elem.name[1]] = all_symbols[elem.name[1]]
      end
      for _, task_symbol in pairs(task_symbols) do
        local task_exprs = terralib.newlist()
        task_exprs:insert(task_symbol)

        local symbols = symbols_by_task[task_symbol]
        local required = {}
        if #elem.classes > 0 then
          elem.classes:map(function(id) required[id] = true end)
        end
        if #elem.constraints > 0 then
          elem.constraints:map(function(id) required[id] = true end)
        end

        for idx, symbol in pairs(all_symbols) do
          if symbols[idx] then
            if required[idx] then
              task_exprs:insert(regex.expr.Concat {
                values = terralib.newlist {
                  symbols[idx],
                  regex.expr.Kleene {
                    value = symbol
                  }
                }
              })
            else
              task_exprs:insert(regex.expr.Kleene {
                value = symbol
              })
            end
          end
        end

        if #task_exprs == 1 then
          elem_exprs:insert(task_exprs[1])
        else
          elem_exprs:insert(regex.expr.Concat {
            values = task_exprs,
          })
        end
      end
      values:insert(regex.expr.Disj { values = elem_exprs })
    end
    return regex.expr.Concat { values = values }
  end)
end

local optimize_match = {}

function optimize_match.mapper(node)
  fetch_task_ids()
  local rules = node.rules
  local selectors, all_symbols, all_task_symbols, symbols_by_task =
    collect_symbols(rules)
  local regexprs =
    translate_to_regex(selectors, all_symbols, all_task_symbols, symbols_by_task)
  --regexprs:map(function(regexpr) print(regex.pretty(regexpr)) end)

  local dfa = automata.product(regexprs:map(automata.regex_to_dfa))
  dfa:renumber()
  local function check(idx, depth)
    local num_task_elements = #rules[idx].selectors[1].elements
    if rules[idx].rule_type == "region" then
      num_task_elements = num_task_elements - 1
    end
    local min_length = num_task_elements
    for eidx = 1, num_task_elements do
      local elem = rules[idx].selectors[1].elements[eidx]
      min_length = min_length + #elem.classes
      min_length = min_length + #elem.constraints
    end

    if num_task_elements > depth then
      assert(false, "error: selector " .. rules[idx].selectors[1]:unparse() ..
      " cannot have a path shorter than " .. tostring(min_length) ..
      " on state digram (length " .. tostring(depth).." found)")
    end
  end
  dfa:verify_tags(check)

  return ast.optimized.Mapper {
    automata = dfa,
    rules = node.rules,
    assignments = node.assignments,
    position = node.position,
  }
end

return optimize_match
