# Realm Machine Model

This example illustrates how to query the machine model from Realm. In addition, it outlines 
various types that Realm uses to define resources of underlying hardware that an application 
occupies, as well as affinities between these resources.

Here is a list of covered topics:

* [Machine](#machine)
* [Processor](#processor)
* [Memory](#memory)
* [Affinity](#affinity)
* [ID](#id)
* [References](#references)

## Machine

In Realm, a Machine is the highest level of machine model, representing all the computer nodes an application can occupy. 
It can be retrieved using the `Machine::get_machine()` function, 
which returns a singleton object.
Processors, memories of the `Machine`, and affinity information can be queried from the `Machine` object.

## Processor

A `Processor` is a runtime object that represents any execution resource that can run a task, such as CPU, GPU, etc.
To retrieve a Processor, `ProcessorQuery` can be used. In this example, we use `ProcessorQuery` to iterate over all 
enabled processors and print out their address space and ID (a unique identifier Realm assignes).
```c++
for(Machine::ProcessorQuery::iterator it = Machine::ProcessorQuery(machine).begin(); it; ++it) {
  ...
}
```
The address space of a Realm resource, such as Processor and Memory (introduced in the next section), indicates the process/rank where it resides.

A `Processor` can also be queried by its `kind` and the runtime supports the following options:

- `TOC_PROC` represents a throughput processor, which is usually a CPU core. 
It can be specified by `-ll:cpu`.
- `LOC_PROC` represents a latency processor (GPU).Currently, Realm supports both NVIDIA and AMD GPUs.
It can be specified by `-ll:gpu`.
- `UTIL_PROC` represents a CPU processor that is designed for users to run their own background work.
It can be specified by `-ll:util`.
- `IO_PROC` represents a processor that is used for I/O, which is also a CPU core. 
It can be specified by `-ll:io`.
- `PROC_GROUP` represents a group of processors. 
- `PROC_SET` represents a set of processors for OpenMP/Kokkos etc. It can be specified by `-ll:mp_nodes`.
- `OMP_PROC` represents OpenMP thread pool. It can be specified by `-ll:ocpu`. The number of threads per `OMP_PROC` can be specified by `-ll:othr`.
- `PY_PROC` represents a CPU processor that is used for Python interpreter. 
It can be specified by `-ll:py`. Currently, we only support a single `PY_PROC`.

For a list of all the processors kind supported by Realm, please refer to [Full Processor Kind](#full-proc-kind).

## Memory

`Memory` is used to describe the location of application data. `MemoryQuery` can be used to query 
the Memory. For example, a `MemoryQuery` can be created with the condition `has_affinity_to` to return 
all memories that are affixed to the given processor.
```c++
Machine::MemoryQuery mq = Machine::MemoryQuery(machine).has_affinity_to(p, 0, 0);
```

A `Memory` can also be queried by its `kind` and the runtime supports the following options:

- `GLOBAL_MEM` represents CPU memory guaranteed to be visible to all processors on all nodes.
e.g. GASNet global memory. `GLOBAL_MEM` is usually slow. `GLOBAL_MEM` is only used by MPI and GASNet1 modules, 
and it can be specified by `-ll:gsize`.
- `SYSTEM_MEM` represents CPU memory visible to all processors on a node.
It can be specified by `-ll:csize`.
- `REGDMA_MEM` represents registered memory visible to all processors on a node, and can be a target of RDMA.
It can be specified by `-ll:rsize`.
- `SOCKET_MEM` represents CPU memory visible to all processors within a node. It is NUMA-aware, so it
provides better performance for processors on the same socket.
- `Z_COPY_MEM` represents Zero-Copy memory visible to all CPUs within a node and one or more GPUs.
It can be specified by `-ll:zsize`.
- `GPU_FB_MEM` represents framebuffer memory for a particular GPU.
It can be specified by `-ll:fsize`.
- `GPU_MANAGED_MEM` represents managed memory that can be cached by either host or GPU.
It can be specified by `-ll:msize`.
- `GPU_DYNAMIC_MEM` represents dynamically-allocated framebuffer memory for a particular GPU.
Its size is not fixed, but its maximum size can be specified by `-cuda:dynfb_max`.
- `DISK_MEM` represents disk memory visible to all processors on a node.
It can be specified by `-ll:dsize`.
- `HDF_MEM` and `FILE_MEM` represent HDF and file memory visible to all processors on a node, respectively. 
They do not have memory space, so their sizes are always 0.
These I/O related memories allow users to create instances for I/O operations. 

For a list of all the memories kind supported by Realm, please refer to [Full Memory Kind](#full-mem-kind).

## Affinity

Realm provides `ProcessorMemoryAffinity` and `MemoryMemoryAffinity` to query the affinity information
between processors and memories. In this example, we use `ProcessorMemoryAffinity` to retrieve the information,
including latency and bandwidth between a pair of memory and processor.
```c++
std::vector<Machine::ProcessorMemoryAffinity> pm_affinity;
machine.get_proc_mem_affinity(pm_affinity, p, m, true/*local_only*/);
unsigned bandwidth = pm_affinity[0].bandwidth;
unsigned latency = pm_affinity[0].latency;
``` 

## ID

Realm ID is a 64-bit value that uniquely encodes both the type of the referred-to Realm object and its identity.
Once we convert an ID into a hexadecimal number, it can be decoded. The following is an example of a processor ID 
and a memory ID:
```
Processor ID 1d00010000000001 is CPU.
System Memory ID 1e00010000000000 has 0 KB, bandwidth 100, latency 5.
```
The highest two digits (8 bits) are used to tell the type of an ID, e.g., `1d` represents Processor and `1e` represents Memory. 
The next four digits (16 bits) are used to tell the owner node of an ID, e.g., `0001` means the processor/memory is on
node 1. The last two/three digits (8/12 bits) are used tell the local index of a Memory/Processor ID, e.g., `01` means
the cpu index is 1 while `00` means the memory index is 0. 
Besides processor and memory, the ID of other Realm objects (e.g., Event and etc.) can also be decoded. 
For a complete introduction to Realm ID, please refer to the [ID header file](#id-header-file).

Realm provides `is_TYPE` functions to test the type of an ID, e.g. in this example, `is_processor` is used to check if an ID 
is a processor.

## References

<div id="full-proc-kind"></div>
[1]: [Full Processor Kind](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/realm_c.h#L45)

<div id="full-mem-kind"></div>
[2]: [Full Memory Kind](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/realm_c.h#L63)

<div id="id-header-file"></div>
[3]. [ID header file](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/id.h)
