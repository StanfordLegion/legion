-- Copyright 2022 Stanford University
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

import "regent"

local format = require("std/format")

local dynamic_array = terralib.memoize(
  function(elt_type)
    local st = terralib.types.newstruct(
      "dynamic_array(" .. tostring(elt_type) .. ")")
    st.entries = terralib.newlist({
        { "size", uint64 },
        { "data", &elt_type },
    })

    local elt_size = terralib.sizeof(elt_type)

    terra st.create(n : uint64)
      var result = st {
        size = n,
        data = [&elt_type](regentlib.c.malloc(n * elt_size)),
      }
      regentlib.assert(
        result.data ~= nil,
        ["malloc failed in " .. tostring(st) .. ".create"])
      return result
    end

    function st:__compute_serialized_size(value_type, value)
      local result = terralib.newsymbol(regentlib.c.size_t, "result")
      local element = terralib.newsymbol(&elt_type)

      local size_actions, size_value = regentlib.compute_serialized_size_inner(
        elt_type, `(@element))
      local actions = quote
        var [result] = 0
        for i = 0, value.size do
          var [element] = value.data + i
          [size_actions]
          result = result + terralib.sizeof(elt_type) + size_value
        end
      end
      return actions, result
    end

    function st:__serialize(value_type, value, fixed_ptr, data_ptr)
      local actions = regentlib.serialize_simple(value_type, value, fixed_ptr, data_ptr)

      local element = terralib.newsymbol(elt_type)
      local element_ptr = terralib.newsymbol(&elt_type)

      local ser_actions = regentlib.serialize_inner(
        elt_type, element, element_ptr, data_ptr)
      return quote
        [actions]
        for i = 0, value.size do
          var [element] = value.data[i]
          var [element_ptr] = [&elt_type](@data_ptr)
          @data_ptr = @data_ptr + terralib.sizeof(elt_type)
          [ser_actions]
        end
      end
    end

    function st:__deserialize(value_type, fixed_ptr, data_ptr)
      local result = terralib.newsymbol(value_type, "result")
      local actions = quote
        var [result] = [regentlib.deserialize_simple(value_type, fixed_ptr, data_ptr)]
      end

      local element_ptr = terralib.newsymbol(&elt_type)

      local deser_actions, deser_value = regentlib.deserialize_inner(
        elt_type, element_ptr, data_ptr)
      actions = quote
        [actions]
        result.data = [&elt_type](regentlib.c.malloc(
          terralib.sizeof(elt_type) * result.size))
        regentlib.assert(result.data ~= nil, "malloc failed in deserialize")
        for i = 0, result.size do
          var [element_ptr] = [&elt_type](@data_ptr)
          @data_ptr = @data_ptr + terralib.sizeof(elt_type)
          [deser_actions]
          result.data[i] = deser_value
        end
      end
      return actions, result
    end

    return st
  end)

local da_int = dynamic_array(int)
local da_string = dynamic_array(regentlib.string)

task pushed(arr : da_int, x : int)
  var result = [da_int.create](arr.size + 1)
  regentlib.c.memcpy(result.data, arr.data, arr.size * [terralib.sizeof(int)])
  result.data[arr.size] = x
  return result
end

task sum_length(arr : da_string)
  var result = 0
  for i = 0, arr.size do
    result += regentlib.c.strlen(arr.data[i])
  end
  return result
end

task main()
  var first = [da_int.create](5)
  for i = 0, 5 do
    first.data[i] = i
  end

  var second = pushed(first, 123)

  format.println("first.size {}", first.size)
  format.println("second.size {}", second.size)
  regentlib.assert(second.size == 6, "test failed")
  regentlib.assert(second.data[4] == 4, "test failed")
  regentlib.assert(second.data[5] == 123, "test failed")

  var third = [da_string.create](3)
  third.data[0] = "asdf"
  third.data[1] = "1"
  third.data[2] = "abc"

  regentlib.assert(sum_length(third) == 8, "test failed")
end
regentlib.start(main)
