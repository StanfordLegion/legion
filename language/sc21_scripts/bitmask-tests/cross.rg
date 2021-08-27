import "regent"
 
local C = regentlib.c
local fmt = require('std/format')
 
local colors = 2e6 -- Double the size of the launch domain
local N = 5 -- Number of arguments
 
__demand(__inline)
task expr_RW(i : int64) return i + (colors / 2) end

__demand(__inline)
task expr_RO(i : int64) return i % (colors / 2) end
 
__demand(__inline)
task check()
  var bitmask : uint8[colors]
  for i = 0, colors do bitmask[i] = 0 end
 
  var value : int64
  var conflict : uint8 = 0

  for i = 0, colors / 2 do
    value = expr_RW(i)
    if (value >= 0) and (value < colors) then
      conflict = bitmask[value]
      bitmask[value] = 1
      if conflict ~= 0 then fmt.println('Error') break end
    end
  end

  for _ = 0, N - 1 do
    for i = 0, colors / 2 do
      value = expr_RO(i)
      if (value >= 0) and (value < colors) then
        conflict = bitmask[value]
        if conflict ~= 0 then fmt.println('Error') break end
      end
    end
  end
end
 
task main()
  var itime = C.legion_get_current_time_in_micros()
  check()
  var ftime = C.legion_get_current_time_in_micros()
  fmt.println("Time (in us): {}", ftime - itime)
end
regentlib.start(main)
