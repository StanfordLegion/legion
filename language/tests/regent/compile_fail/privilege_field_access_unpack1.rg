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

-- fails-with:
-- privilege_field_access_unpack1.rg:57: invalid privilege reads($rw.in_node) for dereference of ptr(Wire($rn), $rw)
--       var i = w.in_node.voltage
--               ^

import "regent"

fspace Currents {
  _0 : float,
  _1 : float,
  _2 : float,
}

fspace Voltages {
  _1 : float,
  _2 : float,
}

fspace Node {
  capacitance : float,
  leakage     : float,
  charge      : float,
  voltage     : float,
}

fspace Wire(rn : region(Node)) {
  in_node     : ptr(Node, rn),
  out_node    : ptr(Node, rn),
  inductance  : float,
  resistance  : float,
  capacitance : float,
  current     : Currents,
  voltage     : Voltages,
}

task calculate_new_currents(rn : region(Node), rw : region(Wire(rn)))
where
  reads(rw.{resistance}, rn.voltage),
  reads writes(rw.{current, voltage})
do
  for w in rw do
    for j = 0, 3 do
      var i = w.in_node.voltage
    end
  end
end
calculate_new_currents:compile()
