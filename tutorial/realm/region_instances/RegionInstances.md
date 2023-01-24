# Region Instances
## Introduction
Realm stores application data in `RegionInstances`. Each
`RegionInstance` is associated with a particular `Memory` and data
stored in this memory cannot be moved. Any data migration in Realm
results in a copy operation.

## Public Interface
```c++
  class REALM_PUBLIC_API RegionInstance {
  public:
    ...
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 InstanceLayoutGeneric *ilg,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    ...
    static Event create_external_instance(RegionInstance& inst,
					  Memory memory,
					  InstanceLayoutGeneric *ilg,
					  const ExternalInstanceResource& resource,
					  const ProfilingRequestSet& prs,
					  Event wait_on = Event::NO_EVENT);
    ...
   }
```

## Instance Layouts
To represent dense data `RegionInstance` can use multi-dimensional
rectangles and bitmasks for sparse unstructured data.

### Accessors
TODO: Discuss AffineAccessor, MultiAffineAccessor, GenericAccessor.

## Example
TODO: Demonstrate AOS/SOA.

```c++
void top_level_task() {
  RegionInstance instance;
  ...
}
```
