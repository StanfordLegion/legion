-- Copyright 2020 Stanford University
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

local cexternal_test
do
  local root_dir = arg[0]:match(".*/") or "./"

  local include_path = ""
  local include_dirs = terralib.newlist()
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local external_test_cc = root_dir .. "external_test.cc"
  local external_test_so
  if os.getenv('SAVEOBJ') == '1' then
    external_test_so = root_dir .. "libexternal_test.so"
  else
    external_test_so = os.tmpname() .. ".so" -- root_dir .. "mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CC_FLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 external_test_cc .. " -o " .. external_test_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. external_test_cc)
    assert(false)
  end
  regentlib.linklibrary(external_test_so)
  cexternal_test =
    terralib.includec("external_test.h", include_dirs)
end

local c = regentlib.c

fspace fs
{
  _0 : float,
  _1 : float,
  _2 : float,
}

task init(r : region(ispace(int2d), fs))
where reads writes(r)
do
  for e in r do
    e._0 = (e.x + 1) * 100 + (e.y + 1) * 10 + 1
    e._1 = (e.x + 1) * 100 + (e.y + 1) * 10 + 2
    e._2 = (e.x + 1) * 100 + (e.y + 1) * 10 + 3
  end
end

terra get_raw_ptr(pr : c.legion_physical_region_t[1],
                  fld : c.legion_field_id_t[1])
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr[0], fld[0])
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = 0
  rect.lo.x[1] = 0
  rect.hi.x[0] = 2
  rect.hi.x[1] = 2
  var p = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return [&float](p)
end

task aos_test_child(r : region(ispace(int2d), fs))
where reads(r)
do
  c.printf("=== child task of aos_test ===\n")
  var p1 = get_raw_ptr(__physical(r._0), __fields(r._0))
  var p2 = get_raw_ptr(__physical(r._1), __fields(r._1))
  var p3 = get_raw_ptr(__physical(r._2), __fields(r._2))
  c.printf("%p %p %p\n", p1, p2, p3)
  var errors = 0
  for i = 0, 27 do
    -- layout not forced by ExternalMapper, defaults to SOA in DefaultMapper
    -- SOA order: fastest..slowest = x, y, field
    var exp_x = i % 3
    var exp_y = (i / 3) % 3
    var exp_f = i / 9
    var exp_val = (exp_x + 1) * 100 + (exp_y + 1) * 10 + (exp_f + 1)
    var act_val = p1[i]
    if(exp_val == act_val) then
      c.printf("%.0f ", act_val)
    else
      c.printf("%.0f(!=%d) ", act_val, exp_val)
      errors += 1
    end
  end
  c.printf("\n")
  regentlib.assert(errors == 0, "mismatches in data")
end

__demand(__external)
task aos_test(r : region(ispace(int2d), fs))
where reads writes(r)
do
  var p1 = get_raw_ptr(__physical(r._0), __fields(r._0))
  var p2 = get_raw_ptr(__physical(r._1), __fields(r._1))
  var p3 = get_raw_ptr(__physical(r._2), __fields(r._2))
  c.printf("=== aos_test ===\n")
  c.printf("%p %p %p\n", p1, p2, p3)
  var errors = 0
  for i = 0, 27 do
    -- AOS order: fastest..slowest = field, x, y
    var exp_f = i % 3
    var exp_x = (i / 3) % 3
    var exp_y = i / 9
    var exp_val = (exp_x + 1) * 100 + (exp_y + 1) * 10 + (exp_f + 1)
    var act_val = p1[i]
    if(exp_val == act_val) then
      c.printf("%.0f ", act_val)
    else
      c.printf("%.0f(!=%d) ", act_val, exp_val)
      errors += 1
    end
  end
  c.printf("\n")
  regentlib.assert(errors == 0, "mismatches in data")
  aos_test_child(r)
end

task soa_test(r : region(ispace(int2d), fs))
where reads writes(r)
do
  var p1 = get_raw_ptr(__physical(r._0), __fields(r._0))
  var p2 = get_raw_ptr(__physical(r._1), __fields(r._1))
  var p3 = get_raw_ptr(__physical(r._2), __fields(r._2))
  c.printf("=== soa_test ===\n")
  c.printf("%p %p %p\n", p1, p2, p3)
  var errors = 0
  for i = 0, 27 do
    -- SOA order: fastest..slowest = x, y, field
    var exp_x = i % 3
    var exp_y = (i / 3) % 3
    var exp_f = i / 9
    var exp_val = (exp_x + 1) * 100 + (exp_y + 1) * 10 + (exp_f + 1)
    var act_val = p1[i]
    if(exp_val == act_val) then
      c.printf("%.0f ", act_val)
    else
      c.printf("%.0f(!=%d) ", act_val, exp_val)
      errors += 1
    end
  end
  c.printf("\n")
  regentlib.assert(errors == 0, "mismatches in data")
end

task toplevel()
  var r = region(ispace(int2d, {3, 3}), fs)
  init(r)
  aos_test(r)
  soa_test(r)
end

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local link_flags = {"-L" .. root_dir, "-lexternal_test"}
  regentlib.saveobj(toplevel, "external_test", "executable", cexternal_test.register_mappers, link_flags)
else
  regentlib.start(toplevel, cexternal_test.register_mappers)
end
