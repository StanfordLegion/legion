-- Copyright 2023 Stanford University
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

task main()
  regentlib.assert(int1d(0) % rect1d { 2, 3 } == int1d(2), "test failed")
  regentlib.assert(int1d(1) % rect1d { 2, 3 } == int1d(3), "test failed")
  regentlib.assert(int1d(2) % rect1d { 2, 3 } == int1d(2), "test failed")
  regentlib.assert(int1d(3) % rect1d { 2, 3 } == int1d(3), "test failed")
  regentlib.assert(int1d(4) % rect1d { 2, 3 } == int1d(2), "test failed")
  regentlib.assert(int1d(5) % rect1d { 2, 3 } == int1d(3), "test failed")

  regentlib.assert(int1d(-2) % rect1d { -1, 2 } == int1d( 2), "test failed")
  regentlib.assert(int1d(-1) % rect1d { -1, 2 } == int1d(-1), "test failed")
  regentlib.assert(int1d( 0) % rect1d { -1, 2 } == int1d( 0), "test failed")
  regentlib.assert(int1d( 1) % rect1d { -1, 2 } == int1d( 1), "test failed")
  regentlib.assert(int1d( 2) % rect1d { -1, 2 } == int1d( 2), "test failed")
  regentlib.assert(int1d( 3) % rect1d { -1, 2 } == int1d(-1), "test failed")

  regentlib.assert(rect1d { 2, 3 } + int1d(2) == rect1d { 4, 5 }, "test failed")
  regentlib.assert(rect1d { 2, 3 } - int1d(2) == rect1d { 0, 1 }, "test failed")
  regentlib.assert(rect1d { 2, 3 } * int1d(2) == rect1d { 4, 6 }, "test failed")
  regentlib.assert(rect1d { 2, 3 } / int1d(2) == rect1d { 1, 1 }, "test failed")
  regentlib.assert(rect1d { 2, 3 } % int1d(2) == rect1d { 0, 1 }, "test failed")
end
regentlib.start(main)
