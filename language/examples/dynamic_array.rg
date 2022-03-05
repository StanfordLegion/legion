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
      return quote end, `(value.size * elt_size)
    end

    function st:__serialize(value_type, value, fixed_ptr, data_ptr)
      return quote
        terralib.attrstore(&(([&st](fixed_ptr)).size), value.size, { align = 1 })
        regentlib.c.memcpy(@data_ptr, value.data, value.size * elt_size)
        @data_ptr = @data_ptr + value.size * elt_size
      end
    end

    function st:__deserialize(value_type, fixed_ptr, data_ptr)
      local result = terralib.newsymbol(value_type, "result")
      local actions = quote
        var [result] = [st.create](
            terralib.attrload(&(([&st](fixed_ptr)).size), { align = 1 }))
        regentlib.c.memcpy(result.data, @data_ptr, result.size * elt_size)
        @data_ptr = @data_ptr + result.size * elt_size
      end
      return actions, result
    end

    return st
  end)

local da_int = dynamic_array(int)

task pushed(arr : da_int, x : int)
  var result = [da_int.create](arr.size + 1)
  regentlib.c.memcpy(result.data, arr.data, arr.size * [terralib.sizeof(int)])
  result.data[arr.size] = x
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
end
regentlib.start(main)
