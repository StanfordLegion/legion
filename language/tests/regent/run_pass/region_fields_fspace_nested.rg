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
calculate_new_currents:compile()
