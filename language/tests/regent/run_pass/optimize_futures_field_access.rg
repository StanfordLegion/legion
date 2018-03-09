import "regent"

fspace t { a : int }

task f()
  return t { a = 2 }
end

task main()
  var x : t
  x.a = 1
  x = f()
  regentlib.assert(x.a == 2, "test failed")
end
regentlib.start(main)
