-- Copyright 2024 Stanford University
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

local format = require("std/format")

__demand(__replicable)
task main()
  var a : int8 = -1
  var b : uint16 = 65535
  var c : int32 = -12345678
  var d : uint64 = 1234
  var e : bool = true
  format.println("Hello {} {} {} {} {} world!", a, b, c, d, e)

  var z : float = 1.23
  var w : double = 3.45
  format.println("Floats: {} {}", z, w)

  var c32 : complex32 = {3, -4}
  var c64 : complex64 = {-5, 6}
  format.println("Complex32 and Complex64: {} | {}", c32, c64)

  format.println("Formatted: {x} {e}", d, 1.234)
  format.println("Padding/Precision: {08x} {.15} {10.3e}", d, 1.234, 3.456)

  var s = "asdf"
  var t = [regentlib.string]("qwer")
  format.println("String: {} {}", s, t)

  var i1 = int1d(1)
  var i2 = int2d { 1, 2 }
  var i3 = int3d { 1, 2, 3 }
  format.println("int1d {}", i1)
  format.println("int2d {}", i2)
  format.println("int3d {}", i3)

  var is = ispace(int2d, { 2, 2 })
  var r = region(is, int)
  for i in is do
    format.println("int2d(is) {}", i)
  end
  for x in r do
    format.println("int2d(r) {}", x)
  end

  format.println("rect2d {}", is.bounds)

  -- Regent's println *DOES NOT* follow C's printf codes, so this
  -- should just print the literal string.
  format.println("%d %f %s %%")

  format.print("a")
  format.print("b")
  format.println("c")

  var size = format.snprint([rawstring](0), 0, "{} {}", 123, 456)
  var buffer = [rawstring](regentlib.c.malloc(size+1))
  format.snprint(buffer, size+1, "{} {}", 123, 456)
  format.println("{}", buffer)
end
regentlib.start(main)
