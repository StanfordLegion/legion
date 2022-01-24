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

local c = regentlib.c

local fcntl = terralib.includec("fcntl.h")
local unistd = terralib.includec("unistd.h")

-- this is an example of using Lua/Terra to implement a "templated" function
function read_region_data(src_t, dst_t)
  local specialized = terra(runtime : c.legion_runtime_t,
			     ctx : c.legion_context_t,
			     pr : c.legion_physical_region_t,
			     fld : c.legion_field_id_t,
			     filename : &int8,
			     offset : uint64)
    var fd = fcntl.open(filename, fcntl.O_RDONLY)
    regentlib.assert(fd >= 0, "failed to open input file")
    
    var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)

    var ispace = c.legion_physical_region_get_logical_region(pr).index_space
    var itr = c.legion_rect_in_domain_iterator_create_1d(c.legion_index_space_get_domain(runtime, ispace))
    while c.legion_rect_in_domain_iterator_valid_1d(itr) do
      var rect = c.legion_rect_in_domain_iterator_get_rect_1d(itr)
      c.legion_rect_in_domain_iterator_step_1d(itr)
      for idx = rect.lo.x[0], rect.hi.x[0] + 1 do
        var pos : c.legion_ptr_t = c.legion_ptr_t { value = idx }
        @[&dst_t](c.legion_accessor_array_1d_ref(fa, pos)) = 0
        var amt = unistd.pread(fd,
          		     [&dst_t](c.legion_accessor_array_1d_ref(fa, pos)),
          		     sizeof(src_t),
          		     offset + sizeof(src_t) * (idx))
        regentlib.assert(amt == sizeof(src_t), "short read!")
      end
    end
    c.legion_rect_in_domain_iterator_destroy_1d(itr)
    unistd.close(fd)
  end
  specialized:compile()
  return specialized
end

helpers = { 
  read_region_data = read_region_data,
  read_ptr_field = read_region_data(int, int64),
  read_float_field = read_region_data(float, float),
}

return helpers
