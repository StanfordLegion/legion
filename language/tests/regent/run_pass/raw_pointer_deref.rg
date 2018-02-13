import "regent"

task main()
  var x : &int = [&int](regentlib.c.malloc([terralib.sizeof(int)]))
  regentlib.assert(x ~= [&int](0), "malloc failed")
  @x = 123
  regentlib.c.printf("x: %d\n", @x)
  regentlib.c.free(x)
end
regentlib.start(main)
