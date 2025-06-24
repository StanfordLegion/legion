#include "realm/runtime_impl.h"
#include "realm/machine_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
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
  }

  void finalize(void) { delete[] nodes; }
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
  {}

  ~MockMemoryImpl() {}

  AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
                                              bool need_alloc_result, bool poisoned,
                                              TimeLimit work_until) override
  {
    return AllocationResult::ALLOC_INSTANT_SUCCESS;
  }

  void release_storage_immediate(RegionInstanceImpl *inst, bool poisoned,
                                 TimeLimit work_until) override
  {}

  void get_bytes(off_t offset, void *dst, size_t size) override {}

  void put_bytes(off_t offset, const void *src, size_t size) override {}

  void *get_direct_ptr(off_t offset, size_t size) override { return nullptr; }
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

  struct ProcessorMemoriesToBeAdded {
    std::vector<MockProcessorInfo> proc_infos;
    std::vector<MockMemoryInfo> mem_infos;
    std::vector<MockProcessorMemoryAffinity> proc_mem_affinities;
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

    machine->update_kind_maps();
  }
};
