import "regent"

task main()
  regentlib.assert("asdf" ~= nil, "test failed")
  regentlib.assert(isnull(nil), "test failed")
end
regentlib.start(main)
