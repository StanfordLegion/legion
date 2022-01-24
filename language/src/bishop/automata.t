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

-- Bishop Automata Library

local std = require("bishop/std")
local ast = require("bishop/ast")
local regex = require("bishop/regex")
local data = require("common/data")

local function epsilon()
  return { epsilon = true }
end

local function is_epsilon(sym)
  return type(sym) == "table" and rawget(sym, "epsilon")
end

local state = {}
state.__index = state

do
  local state_id = 1
  function state.new()
    local s = {
      id = state_id,
      trans = {},
      tags = {},
    }
    setmetatable(s, state)
    state_id = state_id + 1
    return s
  end
end

function state:clear_transition(sym)
  self.trans[sym] = nil
end

function state:add_transition(sym, next_state)
  assert(not self.trans[sym] or self.trans[sym] == next_state)
  self.trans[sym] = next_state
end

local automata = {}
automata.__index = automata

function automata.make(tbl)
  setmetatable(tbl, automata)
  return tbl
end

function automata:epsilon_closure(initial_states)
  local closure = {}
  local states = {}
  for state, _ in pairs(initial_states) do
    states[#states + 1] = state
  end

  while #states > 0 do
    local state = states[#states]
    states[#states] = nil
    closure[state] = true
    for sym, next_state in pairs(state.trans) do
      if is_epsilon(sym) and not closure[next_state] then
        states[#states + 1] = next_state
      end
    end
  end
  return closure
end

local function copy_states(states)
  local tbl = {}
  for state, _ in pairs(states) do
    tbl[state] = true
  end
  return tbl
end

local function cmp_states(state1, state2)
  return state1.id < state2.id
end

local function hash_states(states)
  local arr = {}
  for state, _ in pairs(states) do
    arr[#arr + 1] = state
  end
  table.sort(arr, cmp_states)
  local str = ""
  for idx = 1, #arr do
    str = str .. "." .. tostring(arr[idx].id)
  end
  return str
end

function automata:determinize()
  local closures = {}
  local dfa_states = {}

  local states = { [self.initial] = true }
  local closure = self:epsilon_closure(states)
  local hash = hash_states(closure)
  closures[hash] = closure

  local dfa_initial = state.new()
  dfa_states[hash] = dfa_initial

  local visit_next = { hash }
  local visited = {}
  while #visit_next > 0 do
    local hash = visit_next[#visit_next]
    visit_next[#visit_next] = nil
    visited[hash] = true
    local closure = closures[hash]
    local dfa_state = dfa_states[hash]

    local next_states = {}
    for state, _ in pairs(closure) do
      for sym, next_state in pairs(state.trans) do
        if not is_epsilon(sym) then
          if not next_states[sym] then next_states[sym] = {} end
          next_states[sym][next_state] = true
        end
      end
    end

    local trans = {}
    for sym, next_states in pairs(next_states) do
      trans[sym] = self:epsilon_closure(next_states)
    end

    for sym, closure in pairs(trans) do
      local hash = hash_states(closure)

      if not closures[hash] then
        local dfa_state = state.new()
        closures[hash] = closure
        dfa_states[hash] = dfa_state
        visit_next[#visit_next + 1] = hash
      end

      dfa_state:add_transition(sym, dfa_states[hash])
    end
  end

  local dfa_final = {}
  for state, _ in pairs(self.final) do
    for hash, closure in pairs(closures) do
      if closure[state] then
        dfa_final[dfa_states[hash]] = true
      end
    end
  end

  return automata.make {
    initial = dfa_initial,
    final = dfa_final,
  }
end

function automata.regex_to_nfa(r)
  if r:is(regex.symbol.TaskId) or
     r:is(regex.symbol.Class) or
     r:is(regex.symbol.Constraint) then
    local initial = state.new()
    local final_state = state.new()
    local final = { [final_state] = true }
    initial:add_transition(r.value, final_state)
    return automata.make {
      initial = initial,
      final = final,
    }

  elseif r:is(regex.expr.Concat) then
    local as = r.values:map(function(r) return automata.regex_to_nfa(r) end)
    for i = 2, #as do
      for state, _ in pairs(as[i - 1].final) do
        state:add_transition(epsilon(), as[i].initial)
      end
    end
    return automata.make {
      initial = as[1].initial,
      final = as[#as].final,
    }

  elseif r:is(regex.expr.Disj) then
    local as = r.values:map(function(r) return automata.regex_to_nfa(r) end)
    local initial = state.new()
    local final = {}
    for i = 1, #as do
      initial:add_transition(epsilon(), as[i].initial)
      for state, _ in pairs(as[i].final) do
        final[state] = true
      end
    end
    return automata.make {
      initial = initial,
      final = final,
    }

  elseif r:is(regex.expr.Kleene) then
    local a = automata.regex_to_nfa(r.value)
    for state, _ in pairs(a.final) do
      state:add_transition(epsilon(), a.initial)
      a.initial:add_transition(epsilon(), state)
    end
    return a

  else
    assert(false, "unreachable")
  end
end

function automata.regex_to_dfa(r, mapping)
  return automata.regex_to_nfa(r):determinize()
end

function automata:cache_transitions()
  self.states = {}
  self.trans = {}
  local states = { self.initial }
  while #states > 0 do
    local state = states[#states]
    states[#states] = nil
    self.states[state] = true
    for sym, next_state in pairs(state.trans) do
      self.trans[#self.trans + 1] = {
        src = state,
        dst = next_state,
        sym = sym,
      }
      if not self.states[next_state] then
        states[#states + 1] = next_state
        self.states[next_state] = true
      end
    end
  end
end

function automata.product(dfas)
  local initial_states = {}
  for idx = 1, #dfas do
    initial_states[dfas[idx].initial] = true
    for state, _ in pairs(dfas[idx].final) do
      state.tags[idx] = true
    end
  end
  local tuples = {}
  local product_states = {}
  local product_initial = state.new()
  local hash = hash_states(initial_states)
  product_states[hash] = product_initial
  tuples[hash] = initial_states

  local product_final = {}
  local visit_next = { hash }
  while #visit_next > 0 do
    local hash = visit_next[#visit_next]
    visit_next[#visit_next] = nil

    local tuple = tuples[hash]
    local product_state = product_states[hash]
    for idx = 1, #dfas do
      for state, _ in pairs(dfas[idx].final) do
        if tuple[state] then
          product_final[product_state] = true
          for tag, _ in pairs(state.tags) do
            product_state.tags[tag] = true
          end
        end
      end
    end

    local trans = {}
    for state, _ in pairs(tuple) do
      for sym, next_state in pairs(state.trans) do
        if not trans[sym] then
          trans[sym] = copy_states(tuple)
        end
        trans[sym][state] = nil
        trans[sym][next_state] = true
      end
    end
    for sym, next_tuple in pairs(trans) do
      local next_hash = hash_states(next_tuple)
      if not tuples[next_hash] then
        local product_state = state.new()
        tuples[next_hash] = next_tuple
        product_states[next_hash] = product_state
        visit_next[#visit_next + 1] = next_hash
      end
      product_state:add_transition(sym, product_states[next_hash])
    end
  end

  return automata.make {
    initial = product_initial,
    final = product_final,
  }
end

function automata:unfold_loop_once(state_to_unfold)
  self:cache_transitions()
  local new_states = {}
  local sym_to_self = {}
  for sym, next_state in pairs(state_to_unfold.trans) do
    if next_state == state_to_unfold then
      new_states[sym] = state.new()
      sym_to_self[sym] = true
    end
  end

  for _, new_state in pairs(new_states) do
    for sym, _ in pairs(sym_to_self) do
      new_state:add_transition(sym, new_states[sym])
    end
  end

  local new_trans = {}
  for sym, next_state in pairs(state_to_unfold.trans) do
    if new_states[sym] then
      new_trans[sym] = new_states[sym]
    else
      for _, new_state in pairs(new_states) do
        new_state:add_transition(sym, next_state)
      end
    end
  end

  for state, _ in pairs(self.states) do
    for sym, next_state in pairs(state.trans) do
      if next_state == state_to_unfold then
        state:clear_transition(sym)
        state:add_transition(sym, new_states[sym])
      end
    end
  end
end

function automata:verify_tags(check)
  self:cache_transitions()
  local depths = {}
  local visit_next = { self.initial }
  for state, _ in pairs(self.states) do
    depths[state] = 2^31
  end

  depths[self.initial] = 0
  while #visit_next > 0 do
    local state = visit_next[#visit_next]
    visit_next[#visit_next] = nil
    for sym, next_state in pairs(state.trans) do
      if depths[next_state] > depths[state] + 1 then
        depths[next_state] = depths[state] + 1
        visit_next[#visit_next + 1] = next_state
      end
    end
  end

  for state, _ in pairs(self.states) do
    for tag, _ in pairs(state.tags) do
      check(tag, depths[state])
    end
  end
end

function automata:renumber()
  local count = 1
  local updated = { [self.initial] = true }
  self.initial.id = 0

  local visit_next = { self.initial }
  while #visit_next > 0 do
    local state = visit_next[#visit_next]
    visit_next[#visit_next] = nil
    local to_visit = {}
    for sym, next_state in pairs(state.trans) do
      if not updated[next_state] then
        next_state.id = count
        updated[next_state] = true
        count = count + 1
        visit_next[#visit_next + 1] = next_state
      end
    end
  end
end

function automata:dot(options)
  local filename = options.filename
  local symbol_mapping = options.symbol_mapping
  local tag_mapping = options.tag_mapping
  local state_mapping = options.state_mapping

  self:cache_transitions()
  local dump = print
  local file
  if filename then
    file = io.open(filename, "w")
    dump = function(msg) file:write(msg .. "\n") end
  end
  dump("digraph G {")
  for _, tuple in pairs(self.trans) do
    local label
    if is_epsilon(tuple.sym) then label = "ε"
    elseif symbol_mapping then label = symbol_mapping[tuple.sym]
    else label = tostring(tuple.sym) end
    dump("    " .. tostring(tuple.src.id) .. " -> " .. tostring(tuple.dst.id)
      .. " [ label=\"" .. label .. "\" ];")
  end
  for state, _ in pairs(self.states) do
    local color = ""
    if state == self.initial then
      color = ",fillcolor=\"tomato3\",style=\"filled\""
    elseif self.final[state] then
      color = ",fillcolor=\"slateblue1\",style=\"filled\""
    end
    local label
    if state_mapping then
      label = state_mapping(state)
    else
      label = "state " .. tostring(state.id)
    end
    if tag_mapping then
      for tag, _ in pairs(state.tags) do
        label = label .. "\\n" .. tag_mapping[tag]
      end
    end
    dump("    " .. tostring(state.id) .. " [label=\"" .. label ..
      "\"" .. color .. "]")
  end
  dump("}")
  if filename then file:close() end
end

return automata
