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

class CProcessorAttrBaseTest {
protected:
  void initialize(void)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplMachineModel>();
    runtime_impl->init(1);
  }

  void finalize(void) { runtime_impl->finalize(); }

public:
  void set_processor(realm_processor_t _proc) { proc = _proc; }

protected:
  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl{nullptr};
  realm_processor_t proc{REALM_NO_PROC};
};

// test realm_processor_get_attributes without parameters

class CProcessorGetAttributesTest : public CProcessorAttrBaseTest,
                                    public ::testing::Test {
protected:
  void SetUp() override
  {
    CProcessorAttrBaseTest::initialize();
    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems = {
        {{0, Processor::Kind::LOC_PROC, 0}},
        {{0, Memory::Kind::SYSTEM_MEM, 1024}},
        {{0, 0, 1000, 1}}};
    runtime_impl->setup_mock_proc_mems(procs_mems);
    set_processor(ID::make_processor(0, 0).convert<Processor>());
  }

  void TearDown() override { CProcessorAttrBaseTest::finalize(); }
};

TEST_F(CProcessorGetAttributesTest, NullRuntime)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_KIND};
  uint64_t values[1];
  realm_status_t status = realm_processor_get_attributes(nullptr, proc, attrs, values, 1);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CProcessorGetAttributesTest, InvalidProcessor)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status =
      realm_processor_get_attributes(runtime, REALM_NO_PROC, attrs, values, 1);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
}

TEST_F(CProcessorGetAttributesTest, ValidLocalProcessorButNotAddedToMachine)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  Processor proc = ID::make_processor(0, 10).convert<Processor>();
  realm_status_t status = realm_processor_get_attributes(runtime, proc, attrs, values, 1);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
}

TEST_F(CProcessorGetAttributesTest, ValidRemoteLocalProcessorButNotAddedToMachine)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  Processor proc = ID::make_processor(1, 10).convert<Processor>();
  realm_status_t status = realm_processor_get_attributes(runtime, proc, attrs, values, 1);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
}

TEST_F(CProcessorGetAttributesTest, NullAttributes)
{
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status =
      realm_processor_get_attributes(runtime, proc, nullptr, values, 1);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE);
}

TEST_F(CProcessorGetAttributesTest, NullValues)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_KIND};
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status =
      realm_processor_get_attributes(runtime, proc, attrs, nullptr, 1);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE);
}

TEST_F(CProcessorGetAttributesTest, ZeroNum)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_KIND};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_get_attributes(runtime, proc, attrs, values, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProcessorGetAttributesTest, InvalidAttribute)
{
  realm_processor_attr_t attrs[] = {REALM_PROCESSOR_ATTR_NUM};
  uint64_t values[1];
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_get_attributes(runtime, proc, attrs, values, 1);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE);
}

// test realm_processor_get_attributes with parameters

struct ProcessorGetAttributesTestParam {
  realm_address_space_t address_space;                   // address space of the processor
  realm_processor_kind_t kind;                           // kind of the processor
  std::vector<realm_processor_attr_t> attrs_to_retrieve; // attributes to be retrieved
};

class CProcessorGetAttributesParamTest
  : public CProcessorAttrBaseTest,
    public ::testing::TestWithParam<ProcessorGetAttributesTestParam> {
protected:
  void SetUp() override
  {
    ProcessorGetAttributesTestParam param = GetParam();
    CProcessorAttrBaseTest::initialize();
    attrs_to_retrieve = param.attrs_to_retrieve;
    realm_address_space_t address_space = param.address_space;
    expected_values[REALM_PROCESSOR_ATTR_ADDRESS_SPACE] = address_space;
    realm_processor_kind_t kind = param.kind;
    expected_values[REALM_PROCESSOR_ATTR_KIND] = kind;

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems = {
        {{0, static_cast<Realm::Processor::Kind>(kind), address_space}},
        {{0, Memory::Kind::SYSTEM_MEM, 1024}},
        {{0, 0, 1000, 1}}};
    runtime_impl->setup_mock_proc_mems(procs_mems);
    set_processor(ID::make_processor(address_space, 0).convert<Processor>());
  }

  void TearDown() override { CProcessorAttrBaseTest::finalize(); }

  std::vector<realm_processor_attr_t> attrs_to_retrieve;
  std::map<realm_processor_attr_t, uint64_t> expected_values;
};

TEST_P(CProcessorGetAttributesParamTest, AttributeRetrieval)
{
  std::vector<uint64_t> values(attrs_to_retrieve.size(), 0);
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_get_attributes(
      runtime, proc, attrs_to_retrieve.data(), values.data(), values.size());
  EXPECT_EQ(status, REALM_SUCCESS);
  for(size_t i = 0; i < values.size(); i++) {
    EXPECT_EQ(values[i], expected_values[attrs_to_retrieve[i]]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CProcessorGetAttributesParamTestInstances, CProcessorGetAttributesParamTest,
    ::testing::Values(
        ProcessorGetAttributesTestParam{
            0, LOC_PROC, {REALM_PROCESSOR_ATTR_ADDRESS_SPACE, REALM_PROCESSOR_ATTR_KIND}},
        // test case 2: request REALM_PROCESSOR_ATTR_ADDRESS_SPACE
        ProcessorGetAttributesTestParam{
            0, LOC_PROC, {REALM_PROCESSOR_ATTR_ADDRESS_SPACE}},
        // test case 3: request REALM_PROCESSOR_ATTR_KIND
        ProcessorGetAttributesTestParam{0, LOC_PROC, {REALM_PROCESSOR_ATTR_KIND}}));
