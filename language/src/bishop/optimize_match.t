-- Copyright 2018 Stanford University, NVIDIA Corporation
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
local data = require("common/data")
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
        region_params = {},
      }
      local task_params = v:get_param_symbols()
      local num_reqs = 0
      for idx = 1, #task_params do
        local param_type_in_signature =
          regent_std.as_read(task_params[idx]:gettype())
        if regent_std.is_region(param_type_in_signature) then
          local param_name = task_params[idx]:hasname()
          local req_indices = {}
          local privileges, privilege_field_paths =
            regent_std.find_task_privileges(param_type_in_signature, v)
          for fidx = 1, #privilege_field_paths do
            num_reqs = num_reqs + 1
            local req = {
              privilege = privileges[fidx],
              fields = terralib.newlist(),
              field_ids = terralib.newlist(),
            }
            -- XXX: should be consistent with Regent's numbering
            local field_id = 100
            local fields = privilege_field_paths[fidx]
            for fidx_ = 1, #fields do
              -- XXX: might not be correct with nested fieldspace
              field_id = field_id + 1
              req.fields:insert(fields[fidx_])
              req.field_ids:insert(field_id)
            end
            signature.reqs[num_reqs] = req
            req_indices[num_reqs] = true
          end
          signature.region_params[param_name] = req_indices
        end
      end
      task_signatures[k] = signature
    end
  end
  return task_signatures
end

local function check_rules_against_signatures(rules, task_signatures)
  if std.config["standalone"] then return end
  for ridx = 1, #rules do
    local rule = rules[ridx]
    local num_elements = #rule.selector.elements
    if rule.rule_type == "region" then
      num_elements = num_elements - 1
    end
    for eidx = 1, num_elements do
      local elem = rule.selector.elements[eidx]
      elem.name:map(function(name)
        if not task_signatures[name] then
          log.error(elem, "task '" .. name .. "' does not exist")
        end
      end)
    end
    if rule.rule_type == "region" then
      local task_elem = rule.selector.elements[num_elements]
      local region_elem = rule.selector.elements[num_elements + 1]
      task_elem.name:map(function(task_name)
        region_elem.name:map(function(region_name)
          if not task_signatures[task_name].region_params[region_name] then
            log.error(region_elem, "parameter '" .. region_name ..
              "' either does not exist or have a non-region type")
          end
        end)
      end)
    end
  end
end

local function fetch_call_graph()
  local call_graph = {}
  if std.config["standalone"] then return call_graph end
  local regent_std = require("regent/std")
  local regent_ast = require("regent/ast")
  for k, v in pairs(_G) do
    if regent_std.is_task(v) then
      call_graph[k] = {}
      if v:has_primary_variant() then
        local task_ast = v:get_primary_variant():get_ast()
        local function record_callees(node)
          if (node:is(regent_ast.specialized.expr.Call) or
              node:is(regent_ast.typed.expr.Call)) and
             regent_std.is_task(node.fn.value) then
            call_graph[k][node.fn.value.name:mkstring()] = true
          end
        end
        regent_ast.traverse_node_postorder(record_callees, task_ast)
      end
    end
  end
  local all_callers = {}
  local all_callees = {}
  for caller, callees in pairs(call_graph) do
    all_callers[caller] = true
    for callee, _ in pairs(callees) do
      all_callees[callee] = true
    end
  end
  local toplevel_tasks = {}
  for caller, _ in pairs(all_callers) do
    if not all_callees[caller] then
      toplevel_tasks[caller] = true
    end
  end

  return call_graph, toplevel_tasks
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
  return results
end

local function collect_symbols(rules, task_signatures)
  -- collect all symbol hashes to assign consistent numbers
  local max_task_id = 0
  local all_task_ids = {}
  local num_constraints = 0
  local all_constraint_ids = {}
  local num_classes = 0
  local all_class_ids = {}

  for _, sig in pairs(task_signatures) do
    all_task_ids[sig.task_id] = sig.task_id
    if max_task_id < sig.task_id then
      max_task_id = sig.task_id
    end
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

  -- TODO: need more robust hashing to avoid unexpected collisions
  local all_ids = {}
  for hash, id in pairs(all_task_ids) do
    all_ids[hash] = id
  end
  for hash, id in pairs(all_constraint_ids) do
    all_ids[hash] = id + max_task_id + 1
  end
  for hash, id in pairs(all_class_ids) do
    all_ids[hash] =
      id + max_task_id + num_constraints + 1
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
          symbols_by_task[task_symbol][id] = all_symbols[id]
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
        local constrained = {}
        if #elem.classes > 0 then
          elem.classes:map(function(id) required[id] = true end)
        end
        if #elem.constraints > 0 then
          elem.constraints:map(function(id)
            required[id] = true
            -- If we hit this assertion, there's a collision in the hash values
            assert(all_symbols[id]:is(regex.symbol.Constraint))
            constrained[all_symbols[id].constraint.field] = true
          end)
        end

        for idx, symbol in pairs(all_symbols) do
          local sym = symbols[idx]
          if sym then
            if required[idx] then
              task_exprs:insert(regex.expr.Concat {
                values = terralib.newlist {
                  symbols[idx],
                  regex.expr.Kleene {
                    value = symbol
                  }
                }
              })
            elseif sym:is(regex.symbol.Constraint) and
                   not constrained[sym.constraint.field] then
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
  return rule1.position.linenumber < rule2.position.linenumber
end

local function dump_dfa(dfa, all_symbols, rules, filename)
  local symbol_mapping = {}
  for idx, symbol in pairs(all_symbols) do
    symbol_mapping[idx] = regex.pretty(symbol)
  end
  local tag_mapping = {}
  for idx = 1, #rules do
    local _, filename =
      string.match(rules[idx].position.filename,
                   "(.-)([^\\/]-%.?([^%.\\/]*))$")
    tag_mapping[idx] =
      rules[idx].selector:unparse() .. " (" ..
      filename .. ":" ..
      tostring(rules[idx].position.linenumber) .. ")"
  end
  local function state_mapping(state)
    local label = "state " .. tostring(state.id)
    if state.last_task_symbol then
      label = label .. "\\n" ..
        "current task: " .. state.last_task_symbol.task_name
    end
    return label
  end
  dfa:dot {
    filename = filename,
    symbol_mapping = symbol_mapping,
    tag_mapping = tag_mapping,
    state_mapping = state_mapping,
  }
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

local function add_trivial_loops(dfa, all_task_symbols)
  for id, _ in pairs(all_task_symbols) do
    if not dfa.initial.trans[id] then
      dfa.initial:add_transition(id, dfa.initial)
    end
  end
end

local function record_last_task_symbol(dfa, all_task_symbols)
  local visited = { [dfa.initial] = true }
  local visit_next = { dfa.initial }
  while #visit_next > 0 do
    local state = visit_next[#visit_next]
    visit_next[#visit_next] = nil
    local last_task_symbol = state.last_task_symbol
    for id, next_state in pairs(state.trans) do
      if not visited[next_state] then
        local sym = all_task_symbols[id] or last_task_symbol
        assert(not next_state.last_task_symbol or
               next_state.last_task_symbol == sym)
        next_state.last_task_symbol = sym
        visited[next_state] = true
        visit_next[#visit_next + 1] = next_state
      end
    end
  end
end

local function prune_impossible_transitions(dfa, all_task_symbols, symbols_by_task,
                                            call_graph, toplevel_tasks)
  local visited = { [dfa.initial] = true }
  local visit_next = { dfa.initial }
  while #visit_next > 0 do
    local state = visit_next[#visit_next]
    visit_next[#visit_next] = nil
    local last_task_symbol = state.last_task_symbol
    if last_task_symbol and call_graph[last_task_symbol.task_name] then
      local callees = call_graph[last_task_symbol.task_name]
      local symbol_ids = {}
      for _, sym in pairs(symbols_by_task[last_task_symbol]) do
        symbol_ids[sym.value] = true
      end

      local trans = {}
      for id, next_state in pairs(state.trans) do
        local sym = all_task_symbols[id]
        if not sym and symbol_ids[id] then trans[id] = next_state
        elseif sym and callees[sym.task_name] then trans[id] = next_state end
      end
      state.trans = trans
    elseif not last_task_symbol then
      local trans = {}
      for id, next_state in pairs(state.trans) do
        local sym = all_task_symbols[id]
        if sym and toplevel_tasks[sym.task_name] then
          trans[id] = next_state
        end
      end
      state.trans = trans
    end
    for id, next_state in pairs(state.trans) do
      if not visited[next_state] then
        visited[next_state] = true
        visit_next[#visit_next + 1] = next_state
      end
    end
  end
end

function merge_indistinguishable_states(dfa, all_task_symbols)
  local states = { dfa.initial }
  local visited = { [dfa.initial] = true}
  while #states > 0 do
    local state = states[#states]
    states[#states] = nil

    local closure = {}
    for sym, next_state in pairs(state.trans) do
      if not all_task_symbols[sym] then
        closure[next_state] = sym
      end
    end
    local mergeable = {}
    for next_state, sym in pairs(closure) do
      local all_symbols = {}
      for sym, _ in pairs(state.trans) do all_symbols[sym] = true end
      for sym, _ in pairs(next_state.trans) do all_symbols[sym] = true end
      local indistinguishable = true
      for sym, _ in pairs(all_symbols) do
        if state.trans[sym] and next_state.trans[sym] and
           state.trans[sym].id ~= next_state.trans[sym].id then
          indistinguishable = false
          break
        end
      end
      if indistinguishable then
        mergeable[sym] = next_state
      end
    end
    local trans = {}
    for sym, next_state in pairs(state.trans) do
      if mergeable[sym] ~= next_state then
        trans[sym] = next_state
      end
    end
    state.trans = trans
    for _, next_state in pairs(state.trans) do
      if not visited[next_state] then
        visited[next_state] = true
        states[#states + 1] = next_state
      end
    end
  end
end

local function verify_task_symbols(dfa)
  dfa:cache_transitions()
  for state, _ in pairs(dfa.states) do
    assert(state.last_task_symbol or state == dfa.initial,
           "fatal error: state should either be initial or " ..
           "correspond to one task")
  end
end

local optimize_match = {}

function optimize_match.mapper(node)
  local rules = node.rules
  table.sort(rules, compare_rules)

  local task_signatures = fetch_task_signatures()
  check_rules_against_signatures(rules, task_signatures)

  local selectors, all_symbols, all_task_symbols, symbols_by_task =
    collect_symbols(rules, task_signatures)
  local regexprs =
    translate_to_regex(selectors, all_symbols, all_task_symbols, symbols_by_task)

  local dfa = automata.product(regexprs:map(automata.regex_to_dfa))
  add_trivial_loops(dfa, all_task_symbols)
  dfa:unfold_loop_once(dfa.initial)
  record_last_task_symbol(dfa, all_task_symbols)

  local call_graph, toplevel_tasks = fetch_call_graph()
  prune_impossible_transitions(dfa, all_task_symbols, symbols_by_task,
                               call_graph, toplevel_tasks)

  merge_indistinguishable_states(dfa, all_task_symbols)
  verify_task_symbols(dfa)

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
  if std.config["dump-dfa"] ~= "" then
    dump_dfa(dfa, all_symbols, rules, std.config["dump-dfa"])
  end
  --print_signatures(task_signatures)

  dfa:cache_transitions()
  dfa.all_symbols = all_symbols
  return ast.optimized.Mapper {
    automata = dfa,
    rules = node.rules,
    assignments = node.assignments,
    task_signatures = task_signatures,
  }
end

return optimize_match
