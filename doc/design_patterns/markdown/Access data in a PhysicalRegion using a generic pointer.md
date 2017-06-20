Access data in a PhysicalRegion using a generic pointer

```cpp
  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<1>(elem_rect));
  FieldSpace input_fs = runtime->create_field_space(ctx);
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  RegionRequirement req(input_lr, READ_WRITE, EXCLUSIVE, input_lr);
  req.add_field(FID_X);
  req.add_field(FID_Y);  
  InlineLauncher input_launcher(req);
  PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  input_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, double> acc_x = 
    input_region.get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_y = 
    input_region.get_field_accessor(FID_Y).typeify<double>();

  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++) {
    acc_x.write(DomainPoint::from_point<1>(pir.p), drand48());
    acc_y.write(DomainPoint::from_point<1>(pir.p), drand48());
  }
```
  
