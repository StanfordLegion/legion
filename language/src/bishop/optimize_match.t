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

local function fetch_task_signatures()
  local task_signatures = {}
  if std.config["standalone"] then return task_signatures end
  local regent_std = require("regent/std")
  for k, v in pairs(_G) do
    if regent_std.is_task(v) then
      local signature = {
        task_id = v.taskid:asvalue(),
        reqs = terralib.newlist(),
      }
      local task_params = v.ast.params
      local num_reqs = 0
      local field_id = 100 -- should be consistent with Regent's numbering
      for idx = 1, #task_params do
        local param_type_in_signature =
          regent_std.as_read(task_params[idx].param_type)
        if regent_std.is_region(param_type_in_signature) then
          local privileges, privilege_field_paths =
            regent_std.find_task_privileges(param_type_in_signature,
                                     v:getprivileges(),
                                     v:get_coherence_modes(),
                                     v:get_flags())
          for fidx = 1, #privilege_field_paths do
            num_reqs = num_reqs + 1
            local req = {
              privilege = privileges[fidx],
              fields = terralib.newlist(),
              field_ids = terralib.newlist(),
            }
            local fields = privilege_field_paths[fidx]
            for fidx_ = 1, #fields do
              -- XXX: might not be correct with nested fieldspace
              field_id = field_id + 1
              req.fields:insert(fields[fidx_])
              req.field_ids:insert(field_id)
            end
            signature.reqs[num_reqs] = req
          end
        end
      end
      task_signatures[k] = signature
    end
  end
  return task_signatures
end

local function traverse_element(rules, fn)
  local results = terralib.newlist()
  for ridx = 1, #rules do
    local rule = rules[ridx]
    local elements = terralib.newlist()
    local num_elements = #rule.selector.elements
    if rule.rule_type == "region" then num_elements = num_elements - 1 end
    for eidx = 1, num_elements do
      local r = fn(rule.selector.elements[eidx])
      if r then elements:insert(r) end
    end
    if #elements > 0 then results:insert(elements) end
  end
  if #results > 0 then return results end
end

local function collect_symbols(rules, task_signatures)
  -- collect all symbol hashes to assign consistent numbers
  local num_tasks = 0
  local all_task_ids = {}
  local num_constraints = 0
  local all_constraint_ids = {}
  local num_classes = 0
  local all_class_ids = {}

  for _, sig in pairs(task_signatures) do
    all_task_ids[sig.task_id] = sig.task_id
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

  for name, sig in pairs(task_signatures) do
    all_task_symbols[sig.task_id] = regex.symbol.TaskId {
      value = sig.task_id,
      task_name = name,
    }
    all_symbols[sig.task_id] = all_task_symbols[sig.task_id]
  end

  local selectors = traverse_element(rules, function(elem)
    local name = elem.name:map(function(name)
      return task_signatures[name].task_id
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

local function calculate_specificity(selector)
  local s = { 0, 0, 0, 0 }
  for i = 1, #selector.elements do
    local e = selector.elements[i]
    if #e.name > 0 then s[1] = s[1] + 1 end
    if #e.classes > 0 then s[2] = s[2] + 1 end
    if #e.constraints > 0 then s[3] = s[3] + 1 end
  end
  s[4] = #selector.elements
  return s
end

local function compare_rules(rule1, rule2)
  if rule1 == rule2 then return false end
  assert(rule2:is(rule1.node_type))
  local specificity1 = calculate_specificity(rule1.selector)
  local specificity2 = calculate_specificity(rule2.selector)
  for i = 1, 4 do
    if specificity1[i] < specificity2[i] then
      return true
    elseif specificity1[i] > specificity2[i] then
      return false
    end
  end
  return false
end

local function print_dfa(dfa, all_symbols, rules)
  local symbol_mapping = {}
  for idx, symbol in pairs(all_symbols) do
    symbol_mapping[idx] = regex.pretty(symbol)
  end
  local tag_mapping = {}
  for idx = 1, #rules do
    tag_mapping[idx] = rules[idx].selector:unparse()
  end
  dfa:dot(symbol_mapping, tag_mapping)
end

local function print_signatures(task_signatures)
  for name, sig in pairs(task_signatures) do
    print("task name: " .. name)
    print("  task id: " .. tostring(sig.task_id))
    print("  # region requirements: " .. tostring(#sig.reqs))
    for ridx = 1, #sig.reqs do
      local req = sig.reqs[ridx]
      print("  privilege: " .. tostring(req.privilege))
      for fidx = 1, #req.fields do
        print("    field " .. req.fields[fidx]:mkstring() .. ": " ..
          tostring(req.field_ids[fidx]))
      end
    end
  end
end

local optimize_match = {}

function optimize_match.mapper(node)
  local rules = node.rules
  table.sort(rules, compare_rules)

  local task_signatures = fetch_task_signatures()
  local selectors, all_symbols, all_task_symbols, symbols_by_task =
    collect_symbols(rules, task_signatures)
  local regexprs =
    translate_to_regex(selectors, all_symbols, all_task_symbols, symbols_by_task)

  local dfa = automata.product(regexprs:map(automata.regex_to_dfa))
  dfa:renumber()
  local function check(idx, depth)
    local num_task_elements = #rules[idx].selector.elements
    if rules[idx].rule_type == "region" then
      num_task_elements = num_task_elements - 1
    end
    local min_length = num_task_elements
    for eidx = 1, num_task_elements do
      local elem = rules[idx].selector.elements[eidx]
      min_length = min_length + #elem.classes
      min_length = min_length + #elem.constraints
    end

    if num_task_elements > depth then
      assert(false, "error: selector " .. rules[idx].selector:unparse() ..
      " cannot have a path shorter than " .. tostring(min_length) ..
      " on state digram (length " .. tostring(depth).." found)")
    end
  end
  dfa:verify_tags(check)
  --print_dfa(dfa, all_symbols, rules)
  --print_signatures(task_signatures)

  return ast.optimized.Mapper {
    automata = dfa,
    rules = node.rules,
    assignments = node.assignments,
    task_signatures = task_signatures,
  }
end

return optimize_match
