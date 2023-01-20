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
    reads writes(rw.{current, voltage,in_node,out_node}),
    reads(rn,rw.{inductance,resistance,capacitance})
  do

  for w in rw do
    var I0 : float[3];
    I0[0] = w.current._0;
    I0[1] = w.current._1;
    I0[2] = w.current._2;
  end
end

task main()
  var rn = region(ispace(ptr, 4), Node)
  var rw = region(ispace(ptr, 4), Wire(rn))
  fill(rn.{capacitance, leakage, charge, voltage}, 0.0)
  fill(rw.{inductance, resistance, capacitance, current.{_0, _1, _2}, voltage.{_1, _2}}, 0.0)
  fill(rw.{in_node, out_node}, dynamic_cast(ptr(Node, rn), 0))
  calculate_new_currents(rn, rw)
end
regentlib.start(main)
