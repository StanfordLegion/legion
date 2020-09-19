import "regent"

local format = require("std/format")

fspace fsa {
  x : int,
}

fspace fsb {
  y : fsa,
}

local fp = regentlib.field_path("y", "x")
task main()
  var z : fsb
  z.[fp] = 123
  format.println("{}", z.[fp])
end
regentlib.start(main)
