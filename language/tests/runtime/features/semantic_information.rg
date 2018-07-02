-- Copyright 2018 Stanford University, Los Alamos National Laboratory
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

fspace fs
{
  x : int,
  y : int,
}

local TAG = 12345
local c = regentlib.c

task attach_info(r : region(ispace(int1d), fs))
where reads writes(r)
do
  var lr = __raw(r)
  var is = lr.index_space
  var fs = lr.field_space
  var fields = __fields(r)
  var rt = __runtime()

  var info : int[1]
  var p : &opaque = [&opaque]([&int](info))
  info[0] = 10
  c.legion_logical_region_attach_semantic_information(rt, lr, TAG, p, [sizeof(int)], false)
  info[0] += 1
  c.legion_index_space_attach_semantic_information(rt, is, TAG, p, [sizeof(int)], false)
  info[0] += 1
  c.legion_field_space_attach_semantic_information(rt, fs, TAG, p, [sizeof(int)], false)
  info[0] += 1
  c.legion_field_id_attach_semantic_information(rt, fs, fields[0], TAG, p, [sizeof(int)], false)
  info[0] += 1
  c.legion_field_id_attach_semantic_information(rt, fs, fields[1], TAG, p, [sizeof(int)], false)
end

task retrieve_info(r : region(ispace(int1d), fs))
where reads writes(r)
do
  var lr = __raw(r)
  var is = lr.index_space
  var fs = lr.field_space
  var fields = __fields(r)
  var rt = __runtime()

  var result : (&opaque)[1] result[0] = [&opaque](0)
  var p : &&opaque = [&&opaque](result)

  var size : uint64[1] size[0] = 0
  var q : &uint64 = [&uint64](size)

  c.legion_logical_region_retrieve_semantic_information(rt, lr, TAG, p, q, false, true)
  regentlib.assert(([&int](result[0]))[0] == 10 and size[0] == 4, "test fail")
  c.legion_index_space_retrieve_semantic_information(rt, is, TAG, p, q, false, true)
  regentlib.assert(([&int](result[0]))[0] == 11 and size[0] == 4, "test fail")
  c.legion_field_space_retrieve_semantic_information(rt, fs, TAG, p, q, false, true)
  regentlib.assert(([&int](result[0]))[0] == 12 and size[0] == 4, "test fail")
  c.legion_field_id_retrieve_semantic_information(rt, fs, fields[0], TAG, p, q, false, true)
  regentlib.assert(([&int](result[0]))[0] == 13 and size[0] == 4, "test fail")
  c.legion_field_id_retrieve_semantic_information(rt, fs, fields[1], TAG, p, q, false, true)
  regentlib.assert(([&int](result[0]))[0] == 14 and size[0] == 4, "test fail")
end

task main()
  var r = region(ispace(int1d, 5), fs)
  attach_info(r)
  retrieve_info(r)
end

regentlib.start(main)
