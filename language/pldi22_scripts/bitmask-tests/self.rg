import "regent"
 
local C = regentlib.c
local pow = regentlib.pow(double)
local fmt = require('std/format')
 
local colors = 2e6 + 2 -- Double the size of the launch domain
 
__demand(__inline)
task identity(i : int64) return i end

__demand(__inline)
task linear(i : int64) return 2 * i + 2 end

__demand(__inline)
task modular(i : int64) return (i + 42) % colors end

__demand(__inline)
task quadratic(i : int64) return pow(i, 2) + 4 * i + 2 end

__demand(__inline)
task check()
  var bitmask : uint8[colors]
  for i = 0, colors do bitmask[i] = 0 end
 
  var value : int64
  var conflict : uint8 = 0

  for i = 0, colors / 2 do
    value = identity(i) -- Projection functor
    if (value >= 0) and (value < colors) then
      conflict = bitmask[value]
      bitmask[value] = 1
      if conflict ~= 0 then fmt.println("Error") break end
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
