#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

class CMemoryQueryBaseTest {
protected:
  void initialize(int num_nodes)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplMachineModel>();
    runtime_impl->init(num_nodes);
  }

  void finalize(void) { runtime_impl->finalize(); }

protected:
  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl{nullptr};
};

static realm_status_t REALM_FNPTR count_memory(realm_memory_t m, void *user_data)
{
  size_t *count = (size_t *)(user_data);
  (*count)++;
  return REALM_SUCCESS;
}

// test realm_processor_query_create and realm_processor_query_destroy

class CMemoryQueryCreateDestroyTest : public CMemoryQueryBaseTest,
                                      public ::testing::Test {
protected:
  void SetUp() override { CMemoryQueryBaseTest::initialize(1); }

  void TearDown() override { CMemoryQueryBaseTest::finalize(); }
};

TEST_F(CMemoryQueryCreateDestroyTest, CreateDestroy)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_memory_query_t query;
  realm_status_t status = realm_memory_query_create(runtime, &query);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_NE(query, nullptr);
  status = realm_memory_query_destroy(query);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CMemoryQueryCreateDestroyTest, CreateNullRuntime)
{
  realm_memory_query_t query;
  realm_status_t status = realm_memory_query_create(nullptr, &query);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CMemoryQueryCreateDestroyTest, CreateNullQuery)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_memory_query_create(runtime, nullptr);
  EXPECT_EQ(status, REALM_MEMORY_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CMemoryQueryCreateDestroyTest, DestroyNullQuery)
{
  realm_status_t status = realm_memory_query_destroy(nullptr);
  EXPECT_EQ(status, REALM_MEMORY_QUERY_ERROR_INVALID_QUERY);
}

// test realm_processor_query_restrict_to_kind,
// realm_processor_query_restrict_to_address_space and realm_processor_query_iter without
// parameters

class CMemoryQueryTest : public CMemoryQueryBaseTest, public ::testing::Test {
protected:
  void SetUp() override { CMemoryQueryBaseTest::initialize(1); }

  void TearDown() override
  {
    if(query != nullptr) {
      ASSERT_REALM(realm_memory_query_destroy(query));
    }
    CMemoryQueryBaseTest::finalize();
  }

  realm_memory_query_t query{nullptr};
};

TEST_F(CMemoryQueryTest, RestrictToKindNullQuery)
{
  realm_status_t status = realm_memory_query_restrict_to_kind(nullptr, SYSTEM_MEM);
  EXPECT_EQ(status, REALM_MEMORY_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CMemoryQueryTest, RestrictToKindInvalidKind)
{
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_memory_query_create(runtime, &query));

  realm_status_t status =
      realm_memory_query_restrict_to_kind(query, static_cast<realm_memory_kind_t>(1000));
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY_KIND);
}

TEST_F(CMemoryQueryTest, RestrictToAddressSpaceNullQuery)
{
  realm_status_t status = realm_memory_query_restrict_to_address_space(nullptr, 0);
  EXPECT_EQ(status, REALM_MEMORY_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CMemoryQueryTest, IterNullQuery)
{
  realm_status_t status = realm_memory_query_iter(nullptr, nullptr, nullptr, SIZE_MAX);
  EXPECT_EQ(status, REALM_MEMORY_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CMemoryQueryTest, IterNullCallback)
{
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_memory_query_create(runtime, &query));

  realm_status_t status = realm_memory_query_iter(query, nullptr, nullptr, SIZE_MAX);
  EXPECT_EQ(status, REALM_MEMORY_QUERY_ERROR_INVALID_CALLBACK);
}

TEST_F(CMemoryQueryTest, IterSizeZero)
{
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_memory_query_create(runtime, &query));

  size_t count = 0;
  realm_status_t status = realm_memory_query_iter(query, count_memory, &count, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(count, 0);
}

static realm_status_t REALM_FNPTR callback_returns_error(realm_memory_t m,
                                                         void *user_data)
{
  return REALM_ERROR;
}

TEST_F(CMemoryQueryTest, IterCallbackReturnsError)
{
  // add some memories, otherwise, the callback will not be called
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024}, {1, Memory::Kind::SYSTEM_MEM, 1024}},
          {{0, 0, 1000, 1}, {0, 1, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_memory_query_create(runtime, &query));

  size_t count = 0;
  realm_status_t status =
      realm_memory_query_iter(query, callback_returns_error, &count, SIZE_MAX);
  EXPECT_EQ(status, REALM_ERROR);
  EXPECT_EQ(count, 0);
}

TEST_F(CMemoryQueryTest, IterCallbackIsLimitedBySize)
{
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024}, {1, Memory::Kind::SYSTEM_MEM, 1024}},
          {{0, 0, 1000, 1}, {0, 1, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_memory_query_create(runtime, &query));

  size_t count = 0;
  realm_status_t status = realm_memory_query_iter(query, count_memory, &count, 2);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(count, 2);
}

// test realm_processor_query_restrict_to_kind,
// realm_processor_query_restrict_to_address_space and realm_processor_query_iter with
// parameters

struct MemoryQueryTestBaseParam {
  std::vector<std::pair<Memory::Kind, size_t>> mem_infos;
  realm_address_space_t address_space;
};

class CMemoryQueryParamBaseTest : public CMemoryQueryBaseTest {
protected:
  void initialize(const std::vector<MemoryQueryTestBaseParam> &params)
  {
    int num_nodes = static_cast<int>(params.size());
    CMemoryQueryBaseTest::initialize(num_nodes);
    for(const MemoryQueryTestBaseParam &param : params) {
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems;
      // create mock processor infos
      procs_mems.proc_infos.push_back(
          {0, Processor::Kind::LOC_PROC, param.address_space});
      // create mock memory infos
      unsigned int num_memories = 0;
      for(const std::pair<Memory::Kind, size_t> &mem_info : param.mem_infos) {
        procs_mems.mem_infos.push_back(
            {num_memories, mem_info.first, mem_info.second, param.address_space});
        // create mock processor-memory affinities
        procs_mems.proc_mem_affinities.push_back({0, num_memories, 1000, 1});
        // add to the set of all memories
        Memory mem = ID::make_memory(param.address_space, num_memories).convert<Memory>();
        mems_mapped_by_address_space[param.address_space].insert(mem);
        mems_mapped_by_kind[mem_info.first].insert(mem);
        all_mems.insert(mem);
        num_memories++;
      }
      runtime_impl->setup_mock_proc_mems(procs_mems);
    }

    realm_runtime_t runtime = *runtime_impl;
    ASSERT_REALM(realm_memory_query_create(runtime, &query));
  }

  void finalize(void)
  {
    ASSERT_REALM(realm_memory_query_destroy(query));
    CMemoryQueryBaseTest::finalize();
  }

  realm_memory_query_t query{nullptr};
  std::map<realm_address_space_t, std::set<Memory>> mems_mapped_by_address_space;
  std::map<Memory::Kind, std::set<Memory>> mems_mapped_by_kind;
  std::set<Memory> all_mems;
};

// test realm_memory_query_iter

class CMemoryQueryParamIterTest
  : public CMemoryQueryParamBaseTest,
    public ::testing::TestWithParam<std::vector<MemoryQueryTestBaseParam>> {
protected:
  void SetUp() override
  {
    const std::vector<MemoryQueryTestBaseParam> &params = GetParam();
    CMemoryQueryParamBaseTest::initialize(params);
  }

  void TearDown() override { CMemoryQueryParamBaseTest::finalize(); }
};

struct append_memory_args_t {
  std::vector<realm_memory_t> mems;
};

static realm_status_t append_memory(realm_memory_t m, void *user_data)
{
  append_memory_args_t *args = reinterpret_cast<append_memory_args_t *>(user_data);
  args->mems.push_back(m);
  return REALM_SUCCESS;
}

TEST_P(CMemoryQueryParamIterTest, IterAllMemories)
{
  append_memory_args_t mem_query_args;
  ASSERT_REALM(realm_memory_query_iter(query, append_memory, &mem_query_args, SIZE_MAX));

  EXPECT_EQ(mem_query_args.mems.size(), all_mems.size());
  for(const realm_memory_t mem : mem_query_args.mems) {
    EXPECT_TRUE(all_mems.find(Memory(mem)) != all_mems.end());
  }
}

INSTANTIATE_TEST_SUITE_P(
    CMemoryQueryParamIterTestInstances, CMemoryQueryParamIterTest,
    ::testing::Values(std::vector<MemoryQueryTestBaseParam>{
        MemoryQueryTestBaseParam{
            {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}}, 0},
        MemoryQueryTestBaseParam{
            {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}}, 1}}));

// test realm_memory_query_restrict_to_kind

struct MemoryQueryRestrictToKindTestParam {
  std::vector<MemoryQueryTestBaseParam> params;
  Memory::Kind test_kind;
};

class CMemoryQueryParamRestrictToKindTest
  : public CMemoryQueryParamBaseTest,
    public ::testing::TestWithParam<MemoryQueryRestrictToKindTestParam> {
protected:
  void SetUp() override
  {
    const MemoryQueryRestrictToKindTestParam &param = GetParam();
    CMemoryQueryParamBaseTest::initialize(param.params);
    test_kind = param.test_kind;
  }

  void TearDown() override { CMemoryQueryParamBaseTest::finalize(); }
  Memory::Kind test_kind;
};

TEST_P(CMemoryQueryParamRestrictToKindTest, RestrictToKind)
{
  ASSERT_REALM(realm_memory_query_restrict_to_kind(
      query, static_cast<realm_memory_kind_t>(test_kind)));

  append_memory_args_t mem_query_args;
  ASSERT_REALM(realm_memory_query_iter(query, append_memory, &mem_query_args, SIZE_MAX));

  if(mems_mapped_by_kind.find(test_kind) != mems_mapped_by_kind.end()) {
    EXPECT_EQ(mem_query_args.mems.size(), mems_mapped_by_kind[test_kind].size());
    for(const realm_memory_t mem : mem_query_args.mems) {
      EXPECT_TRUE(mems_mapped_by_kind[test_kind].find(Memory(mem)) !=
                  mems_mapped_by_kind[test_kind].end());
    }
  } else {
    EXPECT_EQ(mem_query_args.mems.size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CMemoryQueryParamRestrictToKindTestInstances, CMemoryQueryParamRestrictToKindTest,
    ::testing::Values(
        // test case 1: two nodes, restrict kind to SYSTEM_MEM, expect to have 2 memories
        MemoryQueryRestrictToKindTestParam{
            std::vector<MemoryQueryTestBaseParam>{
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    0},
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    1}},
            Memory::Kind::SYSTEM_MEM},
        // test case 2: two nodes, restrict kind to Z_COPY_MEM, expect to have no results
        MemoryQueryRestrictToKindTestParam{
            std::vector<MemoryQueryTestBaseParam>{
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    0},
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    1}},
            Memory::Kind::Z_COPY_MEM}));

// test realm_memory_query_restrict_to_address_space

struct MemoryQueryRestrictToAddressSpaceTestParam {
  std::vector<MemoryQueryTestBaseParam> params;
  realm_address_space_t test_address_space;
};

class CMemoryQueryParamRestrictToAddressSpaceTest
  : public CMemoryQueryParamBaseTest,
    public ::testing::TestWithParam<MemoryQueryRestrictToAddressSpaceTestParam> {
protected:
  void SetUp() override
  {
    const MemoryQueryRestrictToAddressSpaceTestParam &param = GetParam();
    CMemoryQueryParamBaseTest::initialize(param.params);
    test_address_space = param.test_address_space;
  }

  void TearDown() override { CMemoryQueryParamBaseTest::finalize(); }
  realm_address_space_t test_address_space;
};

TEST_P(CMemoryQueryParamRestrictToAddressSpaceTest, RestrictToAddressSpace)
{
  ASSERT_REALM(realm_memory_query_restrict_to_address_space(query, test_address_space));

  append_memory_args_t mem_query_args;
  ASSERT_REALM(realm_memory_query_iter(query, append_memory, &mem_query_args, SIZE_MAX));

  if(mems_mapped_by_address_space.find(test_address_space) !=
     mems_mapped_by_address_space.end()) {
    EXPECT_EQ(mem_query_args.mems.size(),
              mems_mapped_by_address_space[test_address_space].size());
    for(const realm_memory_t mem : mem_query_args.mems) {
      EXPECT_TRUE(mems_mapped_by_address_space[test_address_space].find(Memory(mem)) !=
                  mems_mapped_by_address_space[test_address_space].end());
    }
  } else {
    EXPECT_EQ(mem_query_args.mems.size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CMemoryQueryParamRestrictToAddressSpaceTestInstances,
    CMemoryQueryParamRestrictToAddressSpaceTest,
    ::testing::Values(
        // test case 1: two nodes, restrict address space to 0, expect to have 2 memories
        MemoryQueryRestrictToAddressSpaceTestParam{
            std::vector<MemoryQueryTestBaseParam>{
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    0},
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    1}},
            0},
        // test case 2: two nodes, restrict address space to 2, expect to have no results
        MemoryQueryRestrictToAddressSpaceTestParam{
            std::vector<MemoryQueryTestBaseParam>{
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    0},
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 1024}, {Memory::Kind::GPU_FB_MEM, 1024}},
                    1}},
            2})); // has no results

// test realm_memory_query_restrict_to_capacity

struct MemoryQueryRestrictToCapacityTestParam {
  std::vector<MemoryQueryTestBaseParam> params;
  size_t test_capacity;
};

class CMemoryQueryParamRestrictToCapacityTest
  : public CMemoryQueryParamBaseTest,
    public ::testing::TestWithParam<MemoryQueryRestrictToCapacityTestParam> {
protected:
  void SetUp() override
  {
    const MemoryQueryRestrictToCapacityTestParam &param = GetParam();
    CMemoryQueryParamBaseTest::initialize(param.params);
    test_capacity = param.test_capacity;
  }

  void TearDown() override { CMemoryQueryParamBaseTest::finalize(); }
  size_t test_capacity;
};

TEST_P(CMemoryQueryParamRestrictToCapacityTest, RestrictToCapacity)
{
  ASSERT_REALM(realm_memory_query_restrict_by_capacity(query, test_capacity));

  append_memory_args_t mem_query_args;
  ASSERT_REALM(realm_memory_query_iter(query, append_memory, &mem_query_args, SIZE_MAX));

  size_t expected_num_mems = 0;
  realm_runtime_t runtime = *runtime_impl;
  for(const Memory &mem : all_mems) {
    // we need to use the c api because the Memory::capacity() calls get_runtime()
    realm_memory_attr_t attrs[] = {REALM_MEMORY_ATTR_CAPACITY};
    uint64_t values[1];
    ASSERT_REALM(realm_memory_get_attributes(runtime, mem, attrs, values, 1));
    if(values[0] >= test_capacity) {
      expected_num_mems++;
    }
  }
  EXPECT_EQ(mem_query_args.mems.size(), expected_num_mems);
}

INSTANTIATE_TEST_SUITE_P(
    CMemoryQueryParamRestrictToCapacityTestInstances,
    CMemoryQueryParamRestrictToCapacityTest,
    ::testing::Values(
        // test case 1: two nodes, restrict capacity to 128, expect to have 3 memories
        MemoryQueryRestrictToCapacityTestParam{
            std::vector<MemoryQueryTestBaseParam>{
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 128}, {Memory::Kind::GPU_FB_MEM, 256}},
                    0},
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 64}, {Memory::Kind::GPU_FB_MEM, 128}},
                    1}},
            128},
        // test case 2: two nodes, restrict capacity to 1024, expect to have no results
        MemoryQueryRestrictToCapacityTestParam{
            std::vector<MemoryQueryTestBaseParam>{
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 128}, {Memory::Kind::GPU_FB_MEM, 256}},
                    0},
                MemoryQueryTestBaseParam{
                    {{Memory::Kind::SYSTEM_MEM, 64}, {Memory::Kind::GPU_FB_MEM, 128}},
                    1}},
            1024}));