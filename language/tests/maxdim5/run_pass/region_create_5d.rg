import "regent"

task f(r : region(ispace(int5d), double))
where reads(r) do
  var t : double = 0.0
  for i in r do
    t += r[i]
  end
  return t
end

task main()
  var r = region(ispace(int5d, { 2, 2, 2, 2, 2 }), double)
  fill(r, 1)
  var t = f(r)
  regentlib.assert(t == 2 * 2 * 2 * 2 * 2, "test failed")
end
regentlib.start(main)
