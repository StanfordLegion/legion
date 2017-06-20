Create a LogicalRegion with an unstructured IndexSpace 

```cpp
  IndexSpace unstructured_is = runtime->create_index_space(ctx, 1024); 
  printf("Created unstructured index space %x\n", unstructured_is.id);
  {
    IndexAllocator allocator = runtime->create_index_allocator(ctx, 
                                                    unstructured_is);
    ptr_t begin = allocator.alloc(1024);
    assert(!begin.is_null());
    printf("Allocated elements in unstructured "
           "space at ptr_t %d\n", begin.value);
  } 
  FieldSpace fs = runtime->create_field_space(ctx);
  printf("Created field space field space %x\n", fs.get_id());
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    FieldID fida = allocator.allocate_field(sizeof(double), FID_FIELD_A);
    assert(fida == FID_FIELD_A);
    FieldID fidb = allocator.allocate_field(sizeof(int), FID_FIELD_B);
    assert(fidb == FID_FIELD_B);
    printf("Allocated two fields with Field IDs %d and %d\n", fida, fidb);
  }
  LogicalRegion unstructured_lr = 
    runtime->create_logical_region(ctx, unstructured_is, fs);
  printf("Created unstructured logical region (%x,%x,%x)\n",
      unstructured_lr.get_index_space().id, 
      unstructured_lr.get_field_space().get_id(),
      unstructured_lr.get_tree_id());
```
