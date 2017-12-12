import "regent"

fspace fs
{
  field : int8;
  payload : double[1678];
}

task init(r : region(ispace(int1d), fs))
where
  reads writes(r)
do
end

task compute(r : region(ispace(int1d), fs))
where
  reads(r)
do
end

task toplevel()
  var r = region(ispace(int1d, 9998), fs)

  var is = ispace(int1d, 2)
  var p = partition(equal, r, is)

  init(r)
  for c in is do
    compute(p[c])
  end
end

regentlib.start(toplevel)
