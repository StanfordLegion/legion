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

class CMemoryBaseTest {
protected:
  void initialize(void)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplMachineModel>();
    runtime_impl->init(1);
  }

  void finalize(void) { runtime_impl->finalize(); }

  void set_memory(realm_memory_t _mem) { mem = _mem; }

protected:
  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl{nullptr};
  realm_memory_t mem{REALM_NO_MEM};
};

// test realm_memory_get_attributes without parameters

class CMemoryGetAttributesTest : public CMemoryBaseTest, public ::testing::Test {
protected:
  void SetUp() override
  {
    CMemoryBaseTest::initialize();
    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems = {
        {{0, Processor::Kind::LOC_PROC, 0}},
        {{0, Memory::Kind::SYSTEM_MEM, 1024}},
        {{0, 0, 1000, 1}}};
    runtime_impl->setup_mock_proc_mems(procs_mems);
    set_memory(ID::make_memory(0, 0).convert<Memory>());
  }

  void TearDown() override { CMemoryBaseTest::finalize(); }
};

TEST_F(CMemoryGetAttributesTest, NullRuntime)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_KIND};
  uint64_t values[1];
  realm_status_t status = realm_memory_get_attributes(nullptr, mem, attrs, values, 1);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CMemoryGetAttributesTest, InvalidMemory)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status =
      realm_memory_get_attributes(runtime, REALM_NO_MEM, attrs, values, 1);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY);
}

TEST_F(CMemoryGetAttributesTest, ValidLocalMemoryButNotAddedToMachine)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  Memory mem = ID::make_memory(0, 10).convert<Memory>();
  realm_status_t status = realm_memory_get_attributes(runtime, mem, attrs, values, 1);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY);
}

TEST_F(CMemoryGetAttributesTest, ValidRemoteMemoryButNotAddedToMachine)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  Memory mem = ID::make_memory(1, 10).convert<Memory>();
  realm_status_t status = realm_memory_get_attributes(runtime, mem, attrs, values, 1);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY);
}

TEST_F(CMemoryGetAttributesTest, NullAttributes)
{
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_memory_get_attributes(runtime, mem, nullptr, values, 1);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_ATTRIBUTE);
}

TEST_F(CMemoryGetAttributesTest, NullValues)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_KIND};
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_memory_get_attributes(runtime, mem, attrs, nullptr, 1);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_ATTRIBUTE);
}

TEST_F(CMemoryGetAttributesTest, ZeroNum)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_memory_get_attributes(runtime, mem, attrs, values, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CMemoryGetAttributesTest, InvalidAttribute)
{
  realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_NUM};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_memory_get_attributes(runtime, mem, attrs, values, 1);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_ATTRIBUTE);
}

// test realm_processor_get_attributes with parameters

struct MemoryGetAttributesTestParam {
  realm_address_space_t address_space;                // address space of the memory
  realm_memory_kind_t kind;                           // kind of the memory
  size_t capacity;                                    // capacity of the memory
  std::vector<realm_memory_attr_t> attrs_to_retrieve; // attributes to be retrieved
};

class CMemoryGetAttributesParamTest
  : public CMemoryBaseTest,
    public ::testing::TestWithParam<MemoryGetAttributesTestParam> {
protected:
  void SetUp() override
  {
    MemoryGetAttributesTestParam param = GetParam();
    CMemoryBaseTest::initialize();
    attrs_to_retrieve = param.attrs_to_retrieve;
    realm_address_space_t address_space = param.address_space;
    expected_values[REALM_MEMORY_ATTR_ADDRESS_SPACE] = address_space;
    realm_memory_kind_t kind = param.kind;
    expected_values[REALM_MEMORY_ATTR_KIND] = kind;
    size_t capacity = param.capacity;
    expected_values[REALM_MEMORY_ATTR_CAPACITY] = capacity;
    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems = {
        {{0, Processor::Kind::LOC_PROC, address_space}},
        {{0, static_cast<Realm::Memory::Kind>(kind), capacity}},
        {{0, 0, 1000, 1}}};
    runtime_impl->setup_mock_proc_mems(procs_mems);
    set_memory(ID::make_memory(0, 0).convert<Memory>());
  }

  void TearDown() override { CMemoryBaseTest::finalize(); }

  std::vector<realm_memory_attr_t> attrs_to_retrieve;
  std::map<realm_memory_attr_t, uint64_t> expected_values;
};

TEST_P(CMemoryGetAttributesParamTest, AttributeRetrieval)
{
  std::vector<uint64_t> values(attrs_to_retrieve.size());
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_memory_get_attributes(
      runtime, mem, attrs_to_retrieve.data(), values.data(), attrs_to_retrieve.size());
  EXPECT_EQ(status, REALM_SUCCESS);
  for(size_t i = 0; i < attrs_to_retrieve.size(); i++) {
    EXPECT_EQ(values[i], expected_values[attrs_to_retrieve[i]]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CMemoryGetAttributesParamTestInstances, CMemoryGetAttributesParamTest,
    ::testing::Values(
        // test case 1: request all attributes in a single request
        MemoryGetAttributesTestParam{0,
                                     SYSTEM_MEM,
                                     128,
                                     {REALM_MEMORY_ATTR_ADDRESS_SPACE,
                                      REALM_MEMORY_ATTR_KIND,
                                      REALM_MEMORY_ATTR_CAPACITY}},
        // test case 2: request REALM_MEMORY_ATTR_ADDRESS_SPACE
        MemoryGetAttributesTestParam{
            0, SYSTEM_MEM, 128, {REALM_MEMORY_ATTR_ADDRESS_SPACE}},
        // request REALM_MEMORY_ATTR_KIND
        MemoryGetAttributesTestParam{0, SYSTEM_MEM, 128, {REALM_MEMORY_ATTR_KIND}},
        // request REALM_MEMORY_ATTR_CAPACITY
        MemoryGetAttributesTestParam{0, SYSTEM_MEM, 128, {REALM_MEMORY_ATTR_CAPACITY}}));
