-- Copyright 2024 Stanford University
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

local c = regentlib.c
local abs = regentlib.fabs(double)

task daxpy(x : region(ispace(int1d), double),
           y : region(ispace(int1d), double),
           a : float)
where
  reads(x, y), writes(y)
do
  for i in x do
    y[i] = a * x[i]
  end
end

task check(y : region(ispace(int1d), double))
where
  reads(y)
do
  for i in y do
    regentlib.assert(abs(@i - 0.5) < 0.00001, "test failed")
  end
end

terra create_index_space(runtime : c.legion_runtime_t,
                         context : c.legion_context_t,
                         n       : uint64)
  return c.legion_index_space_create(runtime, context, n)
end

terra create_logical_region(runtime : c.legion_runtime_t,
                           context : c.legion_context_t,
                           is      : c.legion_index_space_t,
                           fid     : c.legion_field_id_t)
  var fs = c.legion_field_space_create(runtime, context)
  var alloc = c.legion_field_allocator_create(runtime, context, fs)
  c.legion_field_allocator_allocate_field(alloc, [sizeof(double)], fid)
  c.legion_field_allocator_destroy(alloc)
  return c.legion_logical_region_create(runtime, context, is, fs, false)
end

terra create_logical_partition(runtime : c.legion_runtime_t,
                               context : c.legion_context_t,
                               lr : c.legion_logical_region_t,
                               cs : c.legion_index_space_t)
  var ip = c.legion_index_partition_create_equal(runtime, context,
    lr.index_space, cs, 1, c.AUTO_GENERATE_ID)
  return c.legion_logical_partition_create(runtime, lr, ip)
end

function gen_test(import_ispace, import_region1, import_region2, import_partition1, import_partition2)
  local body = terralib.newlist()

  local n = regentlib.newsymbol(int, "n")
  local cs = regentlib.newsymbol()
  local is = regentlib.newsymbol()
  local x = regentlib.newsymbol()
  local y = regentlib.newsymbol()
  local p = regentlib.newsymbol()
  local q = regentlib.newsymbol()

  body:insert(rquote var [cs] = ispace(int1d, 4) end)

  if import_ispace then
    body:insert(rquote
      var raw_is = create_index_space(__runtime(), __context(), [n])
      var [is] = __import_ispace(int1d, raw_is)
    end)
  else
    body:insert(rquote
      var [is] = ispace(int1d, [n])
    end)
  end

  if import_region1 then
    body:insert(rquote
      var raw_lr1 = create_logical_region(__runtime(), __context(), __raw([is]), 123)
      var [x] = __import_region([is], double, raw_lr1, array(123U))
    end)
  else
    body:insert(rquote
      var [x] = region([is], double)
    end)
  end

  if import_region2 then
    body:insert(rquote
      var raw_lr2 = create_logical_region(__runtime(), __context(), __raw([is]), 456)
      var [y] = __import_region([is], double, raw_lr2, array(456U))
    end)
  else
    body:insert(rquote
      var [y] = region([is], double)
    end)
  end

  if import_partition1 then
    body:insert(rquote
      var raw_lp1 = create_logical_partition(__runtime(), __context(), __raw([x]), __raw([cs]))
      var [p] = __import_partition(disjoint, [x], [cs], raw_lp1)
    end)
  else
    body:insert(rquote
      var [p] = partition(equal, [x], [cs])
    end)
  end

  if import_partition2 then
    body:insert(rquote
      var raw_lp2 = create_logical_partition(__runtime(), __context(), __raw([y]), __raw([cs]))
      var [q] = __import_partition(disjoint, [y], [cs], raw_lp2)
    end)
  else
    body:insert(rquote
      var [q] = partition(equal, [y], [cs])
    end)
  end

  local tsk
  task tsk([n])
    [body];

    fill([x], 1.0)
    fill([y], 0.0)

    for idx = 0, 5 do
      __demand(__index_launch)
      for color in [cs] do
        daxpy([p][color], [q][color], 0.5)
      end
      __demand(__index_launch)
      for color in [cs] do
        check([q][color])
      end
    end
  end
  return tsk
end

local tests = terralib.newlist()
for i = 0, 1 do
  for j = 0, 1 do
    for k = 0, 1 do
      for l = 0, 1 do
        for m = 0, 1 do
          local args = { i == 0, j == 0, k == 0, l == 0, m == 0 }
          tests:insert(gen_test(unpack(args)))
        end
      end
    end
  end
end

task main()
  [tests:map(function(test)
    return rquote test(100) end
  end)]
end

regentlib.start(main)
