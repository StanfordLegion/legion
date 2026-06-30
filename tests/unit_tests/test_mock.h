/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm/runtime_impl.h"
#include "realm/machine_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"
#include <vector>

using namespace Realm;

class MockRuntimeImpl : public RuntimeImpl {
public:
  MockRuntimeImpl(void)
    : RuntimeImpl()
  {}
  ~MockRuntimeImpl()
  {
#ifdef DEBUG_REALM
    event_triggerer.shutdown_work_item();
    copy_analyzer.shutdown_work_item();
#endif
  }

  inline operator realm_runtime_t() noexcept
  {
    return reinterpret_cast<realm_runtime_t>(this);
  }

  inline operator realm_runtime_t() const noexcept
  {
    return reinterpret_cast<realm_runtime_t>(const_cast<MockRuntimeImpl *>(this));
  }

  void init(int _num_nodes)
  {
    assert(machine != nullptr);
    nodes = new Node[_num_nodes];
    num_nodes = _num_nodes;
    my_node_id = 0;
    Realm::Network::max_node_id = _num_nodes - 1;
    Realm::Network::my_node_id = my_node_id;
  }

  void finalize(void)
  {
    delete[] nodes;
    Realm::Network::max_node_id = 0;
    Realm::Network::my_node_id = 0;
  }

  realm_address_space_t my_node_id{0};
};

class MockRuntimeImplWithEventFreeList : public MockRuntimeImpl {
public:
  MockRuntimeImplWithEventFreeList(void)
    : MockRuntimeImpl()
  {}

  void init(int num_nodes = 1)
  {
    MockRuntimeImpl::init(num_nodes);
    local_event_free_list = new LocalEventTableAllocator::FreeList(local_events, 0);
  }

  void finalize(void)
  {
    delete local_event_free_list;
    local_event_free_list = nullptr;
    MockRuntimeImpl::finalize();
  }
};

class MockProcessorImpl : public ProcessorImpl {
public:
  MockProcessorImpl(RuntimeImpl *runtime_impl, Processor _me, Processor::Kind _kind)
    : ProcessorImpl(runtime_impl, _me, _kind, 1)
  {}

  ~MockProcessorImpl() {}

  void enqueue_task(Task *task) override {}
  void enqueue_tasks(Task::TaskList &tasks, size_t num_tasks) override {}
  void spawn_task(Processor::TaskFuncID func_id, const void *args, size_t arglen,
                  const ProfilingRequestSet &reqs, Event start_event,
                  GenEventImpl *finish_event, EventImpl::gen_t finish_gen,
                  int priority) override
  {}

  void add_to_group(ProcessorGroupImpl *group) override {}
  void remove_from_group(ProcessorGroupImpl *group) override {}
};

class MockMemoryImpl : public MemoryImpl {
public:
  MockMemoryImpl(RuntimeImpl *_runtime_impl, Memory _me, size_t _size, MemoryKind _kind,
                 Memory::Kind _lowlevel_kind, NetworkSegment *_segment)
    : MemoryImpl(_runtime_impl, _me, _size, _kind, _lowlevel_kind, _segment)
  {
    buffer.resize(_size);
  }

  ~MockMemoryImpl() {}

  AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
                                              bool need_alloc_result, bool poisoned,
                                              TimeLimit work_until) override
  {
    if(allocated_size + inst->metadata.layout->bytes_used > size) {
      return AllocationResult::ALLOC_INSTANT_FAILURE;
    }
    allocated_size += inst->metadata.layout->bytes_used;
    inst->metadata.inst_offset = allocated_size;
    NodeSet early_reqs;
    inst->metadata.mark_valid(early_reqs);
    return AllocationResult::ALLOC_INSTANT_SUCCESS;
  }

  void release_storage_immediate(RegionInstanceImpl *inst, bool poisoned,
                                 TimeLimit work_until) override
  {
    allocated_size -= inst->metadata.layout->bytes_used;
    inst->metadata.initiate_cleanup(inst->me.id, true);
  }

  void get_bytes(off_t offset, void *dst, size_t size) override {}

  void put_bytes(off_t offset, const void *src, size_t size) override {}

  void *get_direct_ptr(off_t offset, size_t size) override { return nullptr; }

  ExternalInstanceResource *generate_resource_info(RegionInstanceImpl *inst,
                                                   const IndexSpaceGeneric *subspace,
                                                   span<const FieldID> fields,
                                                   bool read_only) override
  {
    return new ExternalMemoryResource(reinterpret_cast<uintptr_t>(buffer.data()),
                                      buffer.size(), read_only);
  }

  bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                          size_t &inst_offset) override
  {
    return true;
  }

  size_t allocated_size{0};
  std::vector<char> buffer;
};

// MockRuntimeImpl for machine model tests

static MemoryImpl::MemoryKind get_memory_kind(Memory::Kind kind)
{
  switch(kind) {
  case Memory::Kind::SYSTEM_MEM:
    return MemoryImpl::MemoryKind::MKIND_SYSMEM;
  case Memory::Kind::GPU_FB_MEM:
    return MemoryImpl::MemoryKind::MKIND_GPUFB;
  case Memory::Kind::GLOBAL_MEM:
    return MemoryImpl::MemoryKind::MKIND_GLOBAL;
  case Memory::Kind::GPU_MANAGED_MEM:
    return MemoryImpl::MemoryKind::MKIND_MANAGED;
  case Memory::Kind::GPU_DYNAMIC_MEM:
    return MemoryImpl::MemoryKind::MKIND_GPUFB;
  case Memory::Kind::Z_COPY_MEM:
    return MemoryImpl::MemoryKind::MKIND_ZEROCOPY;
  default:
    assert(false);
  }
}

class MockRuntimeImplMachineModel : public MockRuntimeImpl {
public:
  MockRuntimeImplMachineModel(void)
    : MockRuntimeImpl()
  {}

  void init(int num_nodes) { MockRuntimeImpl::init(num_nodes); }

  void finalize(void) { MockRuntimeImpl::finalize(); }

  struct MockProcessorInfo {
    unsigned int idx;
    Processor::Kind kind;
    realm_address_space_t address_space;
  };

  struct MockMemoryInfo {
    unsigned int idx;
    Memory::Kind kind;
    size_t size;
    realm_address_space_t address_space;
  };

  struct MockProcessorMemoryAffinity {
    unsigned int proc_idx;
    unsigned int mem_idx;
    unsigned int bandwidth;
    unsigned int latency;
  };

  struct MockMemoryMemoryAffinity {
    unsigned int mem1_idx;
    unsigned int mem2_idx;
    unsigned int bandwidth;
    unsigned int latency;
  };

  struct ProcessorMemoriesToBeAdded {
    std::vector<MockProcessorInfo> proc_infos;
    std::vector<MockMemoryInfo> mem_infos;
    std::vector<MockProcessorMemoryAffinity> proc_mem_affinities;
    std::vector<MockMemoryMemoryAffinity> mem_mem_affinities;
  };

  void setup_mock_proc_mems(const ProcessorMemoriesToBeAdded &procs_mems)
  {
    std::vector<Processor> procs;
    std::vector<Memory> mems;
    // add processors
    for(const MockProcessorInfo &proc_info : procs_mems.proc_infos) {
      Processor proc =
          ID::make_processor(proc_info.address_space, proc_info.idx).convert<Processor>();
      MockProcessorImpl *proc_impl = new MockProcessorImpl(this, proc, proc_info.kind);
      nodes[proc_info.address_space].processors.push_back(proc_impl);
      procs.push_back(proc);
    }

    // add memories
    for(const MockMemoryInfo &mem_info : procs_mems.mem_infos) {
      Memory mem =
          ID::make_memory(mem_info.address_space, mem_info.idx).convert<Memory>();
      MockMemoryImpl *mem_impl =
          new MockMemoryImpl(this, mem, mem_info.size, get_memory_kind(mem_info.kind),
                             mem_info.kind, nullptr);
      nodes[mem_info.address_space].memories.push_back(mem_impl);
      mems.push_back(mem);
    }

    // add processor-memory affinities
    for(const MockProcessorMemoryAffinity &pma_info : procs_mems.proc_mem_affinities) {
      Machine::ProcessorMemoryAffinity pma;
      pma.p = procs[pma_info.proc_idx];
      pma.m = mems[pma_info.mem_idx];
      pma.bandwidth = pma_info.bandwidth;
      pma.latency = pma_info.latency;
      add_proc_mem_affinity(pma);
    }

    // add memory-memory affinities
    for(const MockMemoryMemoryAffinity &mma_info : procs_mems.mem_mem_affinities) {
      Machine::MemoryMemoryAffinity mma;
      mma.m1 = mems[mma_info.mem1_idx];
      mma.m2 = mems[mma_info.mem2_idx];
      mma.bandwidth = mma_info.bandwidth;
      mma.latency = mma_info.latency;
      MachineNodeInfo *node_info =
          machine->nodeinfos[mems[mma_info.mem1_idx].address_space()];
      node_info->add_memory(mems[mma_info.mem1_idx]);
      node_info->add_memory(mems[mma_info.mem2_idx]);
      node_info->add_mem_mem_affinity(mma);
    }

    machine->update_kind_maps();
  }
};
