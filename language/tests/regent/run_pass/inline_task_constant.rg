import "regent"

__demand(__inline)
task inc(x : int)
  x += 1
  return x
end

task toplevel()
  var i1 = inc(1)
  var i2 = inc(10)
  var c1 = __forbid(__inline, inc(1))
  var c2 = __forbid(__inline, inc(10))
  regentlib.assert(i1 == c1, "test failed")
  regentlib.assert(i2 == c2, "test failed")
  
end

regentlib.start(toplevel)
