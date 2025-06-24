#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <map>
#include <set>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

struct ProcessorToBeAdded {
  Realm::Processor proc;
  Realm::Processor::Kind kind;
  realm_address_space_t address_space;
};

class MockProcessorImplTaskTest : public MockProcessorImpl {
public:
  MockProcessorImplTaskTest(RuntimeImpl *runtime_impl, Processor _me,
                            Processor::Kind _kind)
    : MockProcessorImpl(runtime_impl, _me, _kind)
  {}

  void spawn_task(Processor::TaskFuncID func_id, const void *args, size_t arglen,
                  const ProfilingRequestSet &reqs, Event start_event,
                  GenEventImpl *finish_event, EventImpl::gen_t finish_gen,
                  int priority) override
  {
    spawned_func_ids.push_back(func_id);
  }
  std::vector<Processor::TaskFuncID> spawned_func_ids;
};

class CProcessorTaskBaseTest {
protected:
  void initialize(void)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithEventFreeList>();
    runtime_impl->init(1);
  }

  void finalize(void) { runtime_impl->finalize(); }

public:
  void setup_mock_processors(const ProcessorToBeAdded &processor)
  {
    // The processors are added to the machine model via add_proc_mem_affinity, so we need
    // a memory
    Memory mem = ID::make_memory(processor.address_space, 0).convert<Memory>();
    MockMemoryImpl *mem_impl =
        new MockMemoryImpl(runtime_impl.get(), mem, 1024, MemoryImpl::MKIND_SYSMEM,
                           Memory::Kind::SYSTEM_MEM, nullptr);
    runtime_impl->nodes[processor.address_space].memories.push_back(mem_impl);

    proc_impl =
        new MockProcessorImplTaskTest(runtime_impl.get(), processor.proc, processor.kind);
    runtime_impl->nodes[processor.address_space].processors.push_back(proc_impl);

    Machine::ProcessorMemoryAffinity pma;
    pma.p = processor.proc;
    pma.m = mem;
    pma.bandwidth = 1000;
    pma.latency = 1;
    runtime_impl->add_proc_mem_affinity(pma);

    // we call it for the local processors
    runtime_impl->machine->update_kind_maps();
  }

protected:
  std::unique_ptr<MockRuntimeImplWithEventFreeList> runtime_impl{nullptr};
  MockProcessorImplTaskTest *proc_impl{nullptr};
};

// test realm_processor_get_attributes without parameters

class CProcessorTaskTest : public CProcessorTaskBaseTest, public ::testing::Test {
protected:
  void SetUp() override
  {
    CProcessorTaskBaseTest::initialize();
    ProcessorToBeAdded processor = {ID::make_processor(0, 0).convert<Processor>(),
                                    Realm::Processor::Kind::LOC_PROC, 0};
    CProcessorTaskBaseTest::setup_mock_processors(processor);
  }

  void TearDown() override { CProcessorTaskBaseTest::finalize(); }
};

TEST_F(CProcessorTaskTest, SpawnNullRuntime)
{
  realm_event_t event;
  realm_status_t status = realm_processor_spawn(nullptr, proc_impl->me, 0, nullptr, 0,
                                                nullptr, REALM_NO_EVENT, 0, &event);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CProcessorTaskTest, SpawnInvalidProcessor)
{
  realm_event_t event;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_spawn(runtime, REALM_NO_PROC, 0, nullptr, 0,
                                                nullptr, REALM_NO_EVENT, 0, &event);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
}

TEST_F(CProcessorTaskTest, SpawnSuccess)
{
  realm_event_t event;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_spawn(runtime, proc_impl->me, 0, nullptr, 0,
                                                nullptr, REALM_NO_EVENT, 0, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(proc_impl->spawned_func_ids.size(), 1);
  EXPECT_EQ(proc_impl->spawned_func_ids[0], 0);
}

static void test_task_func(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, realm_processor_t proc_id)
{}

TEST_F(CProcessorTaskTest, RegisterTaskNullRuntime)
{
  realm_event_t event;
  realm_status_t status = realm_processor_register_task_by_kind(
      nullptr, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, 0, test_task_func, nullptr, 0,
      &event);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CProcessorTaskTest, RegisterTaskInvalidProcessorKind)
{
  realm_event_t event;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_register_task_by_kind(
      runtime, static_cast<realm_processor_kind_t>(100), REALM_REGISTER_TASK_DEFAULT, 0,
      test_task_func, nullptr, 0, &event);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR_KIND);
}

TEST_F(CProcessorTaskTest, RegisterTaskInvalidTaskFunction)
{
  realm_event_t event;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, 0, nullptr, nullptr, 0, &event);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_TASK_FUNCTION);
}

// TODO: test the register task success
