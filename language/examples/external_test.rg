-- Copyright 2017 Stanford University
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
  assert(os.getenv('LG_RT_DIR') ~= nil, "$LG_RT_DIR should be set!")
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local legion_dir = runtime_dir .. "legion/"
  local mapper_dir = runtime_dir .. "mappers/"
  local realm_dir = runtime_dir .. "realm/"
  local external_test_cc = root_dir .. "external_test.cc"
  local external_test_so
  if os.getenv('SAVEOBJ') == '1' then
    external_test_so = root_dir .. "libexternal_test.so"
  else
    external_test_so = os.tmpname() .. ".so" -- root_dir .. "mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " ..
                 " -I " .. realm_dir .. " " .. external_test_cc .. " -o " .. external_test_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. external_test_cc)
    assert(false)
  end
  terralib.linklibrary(external_test_so)
  cexternal_test =
    terralib.includec("external_test.h", {"-I", root_dir, "-I", runtime_dir,
                                          "-I", mapper_dir, "-I", legion_dir,
                                          "-I", realm_dir})
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
  var cnt = 0
  for e in r do
    e._0 = 3 * cnt
    e._1 = 3 * cnt + 1
    e._2 = 3 * cnt + 2
    cnt += 1
  end
end

terra get_raw_ptr(pr : c.legion_physical_region_t[1],
                  fld : c.legion_field_id_t[1])
  var fa = c.legion_physical_region_get_field_accessor_generic(pr[0], fld[0])
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = 0
  rect.lo.x[1] = 0
  rect.hi.x[0] = 3
  rect.hi.x[1] = 3
  var p = c.legion_accessor_generic_raw_rect_ptr_2d(fa, rect, &subrect, offsets)
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
  for i = 0, 27 do
    c.printf("%.0f ", p1[i])
  end
  c.printf("\n")
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
  for i = 0, 27 do
    c.printf("%.0f ", p1[i])
  end
  c.printf("\n")
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
  for i = 0, 27 do
    c.printf("%.0f ", p1[i])
  end
  c.printf("\n")
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
