Create a LogicalRegion with a structured IndexSpace 

```cpp
      typedef float PixelField;
      enum FieldIDs {  
        FID_FIELD_R = 0,
        FID_FIELD_G,
        FID_FIELD_B,
        FID_FIELD_A,
        FID_FIELD_Z,
        FID_FIELD_USERDATA,
      };

      Point<3> origin = Point<3>::ZEROES();
      Point<3> regionSize;
      // HDTV imgae resolution
      regionSize.x[0] = 1920;
      regionSize.x[1] = 1080;
      regionSize.x[2] = numSimulationNodes;
      Rect<3> imageBounds(origin, regionSize - Point<3>::ONES());
      Domain imageDomain = Domain::from_rect<3>(imageBounds);
      pixels = runtime->create_index_space(context, imageDomain);
      runtime->attach_name(pixels, "image index space");
      
      FieldSpace fields = runtime->create_field_space(context);
      runtime->attach_name(fields, "pixel fields");
      {
        FieldAllocator allocator = runtime->create_field_allocator(context, fields);
        FieldID fidr = allocator.allocate_field(sizeof(PixelField), FID_FIELD_R);
        assert(fidr == FID_FIELD_R);
        FieldID fidg = allocator.allocate_field(sizeof(PixelField), FID_FIELD_G);
        assert(fidg == FID_FIELD_G);
        FieldID fidb = allocator.allocate_field(sizeof(PixelField), FID_FIELD_B);
        assert(fidb == FID_FIELD_B);
        FieldID fida = allocator.allocate_field(sizeof(PixelField), FID_FIELD_A);
        assert(fida == FID_FIELD_A);
        FieldID fidz = allocator.allocate_field(sizeof(PixelField), FID_FIELD_Z);
        assert(fidz == FID_FIELD_Z);
        FieldID fidUserdata = allocator.allocate_field(sizeof(PixelField), FID_FIELD_USERDATA);
        assert(fidUserdata == FID_FIELD_USERDATA);
      }
      
      region = runtime->create_logical_region(context, pixels, fields);
      runtime->attach_name(region, "image");
```