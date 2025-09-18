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

class CProcessorQueryBaseTest {
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

static realm_status_t REALM_FNPTR count_processor(realm_processor_t p, void *user_data)
{
  size_t *count = (size_t *)(user_data);
  (*count)++;
  return REALM_SUCCESS;
}

// test realm_processor_query_create and realm_processor_query_destroy

class CProcessorQueryCreateDestroyTest : public CProcessorQueryBaseTest,
                                         public ::testing::Test {
protected:
  void SetUp() override { CProcessorQueryBaseTest::initialize(1); }

  void TearDown() override { CProcessorQueryBaseTest::finalize(); }
};

TEST_F(CProcessorQueryCreateDestroyTest, CreateDestroy)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_processor_query_t query;
  realm_status_t status = realm_processor_query_create(runtime, &query);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_NE(query, nullptr);
  status = realm_processor_query_destroy(query);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProcessorQueryCreateDestroyTest, CreateNullRuntime)
{
  realm_processor_query_t query;
  realm_status_t status = realm_processor_query_create(nullptr, &query);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CProcessorQueryCreateDestroyTest, CreateNullQuery)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_processor_query_create(runtime, nullptr);
  EXPECT_EQ(status, REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CProcessorQueryCreateDestroyTest, DestroyNullQuery)
{
  realm_status_t status = realm_processor_query_destroy(nullptr);
  EXPECT_EQ(status, REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY);
}

// test realm_processor_query_restrict_to_kind,
// realm_processor_query_restrict_to_address_space and realm_processor_query_iter without
// parameters

class CProcessorQueryTest : public CProcessorQueryBaseTest, public ::testing::Test {
protected:
  void SetUp() override { CProcessorQueryBaseTest::initialize(1); }

  void TearDown() override
  {
    if(query != nullptr) {
      ASSERT_REALM(realm_processor_query_destroy(query));
    }
    CProcessorQueryBaseTest::finalize();
  }

  realm_processor_query_t query{nullptr};
};

TEST_F(CProcessorQueryTest, RestrictToKindNullQuery)
{
  realm_status_t status = realm_processor_query_restrict_to_kind(nullptr, LOC_PROC);
  EXPECT_EQ(status, REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CProcessorQueryTest, RestrictToKindInvalidKind)
{
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_processor_query_create(runtime, &query));

  realm_status_t status = realm_processor_query_restrict_to_kind(
      query, static_cast<realm_processor_kind_t>(1000));
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR_KIND);
}

TEST_F(CProcessorQueryTest, RestrictToAddressSpaceNullQuery)
{
  realm_status_t status = realm_processor_query_restrict_to_address_space(nullptr, 0);
  EXPECT_EQ(status, REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CProcessorQueryTest, IterNullQuery)
{
  realm_status_t status = realm_processor_query_iter(nullptr, nullptr, nullptr, SIZE_MAX);
  EXPECT_EQ(status, REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY);
}

TEST_F(CProcessorQueryTest, IterNullCallback)
{
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_processor_query_create(runtime, &query));

  realm_status_t status = realm_processor_query_iter(query, nullptr, nullptr, SIZE_MAX);
  EXPECT_EQ(status, REALM_PROCESSOR_QUERY_ERROR_INVALID_CALLBACK);
}

TEST_F(CProcessorQueryTest, IterSizeZero)
{
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_processor_query_create(runtime, &query));

  size_t count = 0;
  realm_status_t status = realm_processor_query_iter(query, count_processor, &count, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(count, 0);
}

static realm_status_t REALM_FNPTR callback_returns_error(realm_processor_t p,
                                                         void *user_data)
{
  return REALM_ERROR;
}

TEST_F(CProcessorQueryTest, IterCallbackReturnsError)
{
  // add some processors, otherwise, the callback will not be called
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}, {1, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024}},
          {{0, 0, 1000, 1}, {1, 0, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_processor_query_create(runtime, &query));

  size_t count = 0;
  realm_status_t status =
      realm_processor_query_iter(query, callback_returns_error, &count, SIZE_MAX);
  EXPECT_EQ(status, REALM_ERROR);
  EXPECT_EQ(count, 0);
}

TEST_F(CProcessorQueryTest, IterCallbackIsLimitedBySize)
{
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0},
           {1, Processor::Kind::LOC_PROC, 0},
           {2, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024}},
          {{0, 0, 1000, 1}, {1, 0, 1000, 1}, {2, 0, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_processor_query_create(runtime, &query));

  // we added 3 processors, so this should return 2
  size_t count = 0;
  realm_status_t status = realm_processor_query_iter(query, count_processor, &count, 2);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(count, 2);
}

// test realm_processor_query_restrict_to_kind,
// realm_processor_query_restrict_to_address_space and realm_processor_query_iter with
// parameters

struct ProcessorQueryTestBaseParam {
  std::vector<Processor::Kind> kinds;
  realm_address_space_t address_space;
};

class CProcessorQueryParamBaseTest : public CProcessorQueryBaseTest {
protected:
  void initialize(const std::vector<ProcessorQueryTestBaseParam> &params)
  {
    int num_nodes = static_cast<int>(params.size());
    CProcessorQueryBaseTest::initialize(num_nodes);
    for(const ProcessorQueryTestBaseParam &param : params) {
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems;
      // create mock processor infos
      procs_mems.mem_infos.push_back(
          {0, Memory::Kind::SYSTEM_MEM, 1024, param.address_space});
      // create mock memory infos
      unsigned int num_processors = 0;
      for(const Processor::Kind &kind : param.kinds) {
        procs_mems.proc_infos.push_back({num_processors, kind, param.address_space});
        // create mock processor-memory affinities
        procs_mems.proc_mem_affinities.push_back({num_processors, 0, 1000, 1});
        // add to the set of all memories
        Processor proc =
            ID::make_processor(param.address_space, num_processors).convert<Processor>();
        procs_mapped_by_address_space[param.address_space].insert(proc);
        procs_mapped_by_kind[kind].insert(proc);
        all_procs.insert(proc);
        num_processors++;
      }
      runtime_impl->setup_mock_proc_mems(procs_mems);
    }

    realm_runtime_t runtime = *runtime_impl;
    ASSERT_REALM(realm_processor_query_create(runtime, &query));
  }

  void finalize(void)
  {
    ASSERT_REALM(realm_processor_query_destroy(query));
    CProcessorQueryBaseTest::finalize();
  }

  realm_processor_query_t query{nullptr};
  std::map<realm_address_space_t, std::set<Processor>> procs_mapped_by_address_space;
  std::map<Processor::Kind, std::set<Processor>> procs_mapped_by_kind;
  std::set<Processor> all_procs;
};

// test realm_processor_query_iter

class CProcessorQueryParamIterTest
  : public CProcessorQueryParamBaseTest,
    public ::testing::TestWithParam<std::vector<ProcessorQueryTestBaseParam>> {
protected:
  void SetUp() override
  {
    const std::vector<ProcessorQueryTestBaseParam> &params = GetParam();
    CProcessorQueryParamBaseTest::initialize(params);
  }

  void TearDown() override { CProcessorQueryParamBaseTest::finalize(); }
};

struct append_process_args_t {
  std::vector<realm_processor_t> procs;
};

static realm_status_t append_process(realm_processor_t p, void *user_data)
{
  append_process_args_t *args = reinterpret_cast<append_process_args_t *>(user_data);
  args->procs.push_back(p);
  return REALM_SUCCESS;
}

TEST_P(CProcessorQueryParamIterTest, IterAllProcessors)
{
  append_process_args_t proc_query_args;
  ASSERT_REALM(
      realm_processor_query_iter(query, append_process, &proc_query_args, SIZE_MAX));

  EXPECT_EQ(proc_query_args.procs.size(), all_procs.size());
  for(const realm_processor_t proc : proc_query_args.procs) {
    EXPECT_TRUE(all_procs.find(Processor(proc)) != all_procs.end());
  }
}

INSTANTIATE_TEST_SUITE_P(
    CProcessorQueryParamIterTestInstances, CProcessorQueryParamIterTest,
    ::testing::Values(
        // Two nodes, one with 4 processors, and the other with 3 processors
        std::vector<ProcessorQueryTestBaseParam>{
            ProcessorQueryTestBaseParam{
                {Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                 Processor::Kind::TOC_PROC, Processor::Kind::UTIL_PROC},
                0},
            ProcessorQueryTestBaseParam{{Processor::Kind::LOC_PROC,
                                         Processor::Kind::LOC_PROC,
                                         Processor::Kind::OMP_PROC},
                                        1}}));

// test realm_processor_query_restrict_to_kind

struct ProcessorQueryRestrictToKindTestParam {
  std::vector<ProcessorQueryTestBaseParam> params;
  Processor::Kind test_kind;
};

class CProcessorQueryParamRestrictToKindTest
  : public CProcessorQueryParamBaseTest,
    public ::testing::TestWithParam<ProcessorQueryRestrictToKindTestParam> {
protected:
  void SetUp() override
  {
    const ProcessorQueryRestrictToKindTestParam &param = GetParam();
    CProcessorQueryParamBaseTest::initialize(param.params);
    test_kind = param.test_kind;
  }

  void TearDown() override { CProcessorQueryParamBaseTest::finalize(); }
  Processor::Kind test_kind;
};

TEST_P(CProcessorQueryParamRestrictToKindTest, RestrictToKind)
{
  ASSERT_REALM(realm_processor_query_restrict_to_kind(
      query, static_cast<realm_processor_kind_t>(test_kind)));

  append_process_args_t proc_query_args;
  ASSERT_REALM(
      realm_processor_query_iter(query, append_process, &proc_query_args, SIZE_MAX));

  if(procs_mapped_by_kind.find(test_kind) != procs_mapped_by_kind.end()) {
    EXPECT_EQ(proc_query_args.procs.size(), procs_mapped_by_kind[test_kind].size());
    for(const realm_processor_t proc : proc_query_args.procs) {
      EXPECT_TRUE(procs_mapped_by_kind[test_kind].find(Processor(proc)) !=
                  procs_mapped_by_kind[test_kind].end());
    }
  } else {
    EXPECT_EQ(proc_query_args.procs.size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CProcessorQueryParamRestrictToKindTestInstances,
    CProcessorQueryParamRestrictToKindTest,
    ::testing::Values(
        // test case 1: two nodes, restrict kind to LOC_PROC, expect to have 4 processors
        ProcessorQueryRestrictToKindTestParam{
            std::vector<ProcessorQueryTestBaseParam>{
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::TOC_PROC, Processor::Kind::UTIL_PROC},
                 0},
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::OMP_PROC},
                 1}},
            Processor::Kind::LOC_PROC},
        // test case 2: two nodes, restrict kind to PY_PROC, expect to have no results
        ProcessorQueryRestrictToKindTestParam{
            std::vector<ProcessorQueryTestBaseParam>{
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::TOC_PROC, Processor::Kind::UTIL_PROC},
                 0},
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::OMP_PROC},
                 1}},
            Processor::Kind::PY_PROC}));

// test realm_processor_query_restrict_to_address_space

struct ProcessorQueryRestrictToAddressSpaceTestParam {
  std::vector<ProcessorQueryTestBaseParam> params;
  realm_address_space_t test_address_space;
};

class CProcessorQueryParamRestrictToAddressSpaceTest
  : public CProcessorQueryParamBaseTest,
    public ::testing::TestWithParam<ProcessorQueryRestrictToAddressSpaceTestParam> {
protected:
  void SetUp() override
  {
    const ProcessorQueryRestrictToAddressSpaceTestParam &param = GetParam();
    CProcessorQueryParamBaseTest::initialize(param.params);
    test_address_space = param.test_address_space;
  }

  void TearDown() override { CProcessorQueryParamBaseTest::finalize(); }
  realm_address_space_t test_address_space;
};

TEST_P(CProcessorQueryParamRestrictToAddressSpaceTest, RestrictToAddressSpace)
{
  ASSERT_REALM(
      realm_processor_query_restrict_to_address_space(query, test_address_space));

  append_process_args_t proc_query_args;
  ASSERT_REALM(
      realm_processor_query_iter(query, append_process, &proc_query_args, SIZE_MAX));

  if(procs_mapped_by_address_space.find(test_address_space) !=
     procs_mapped_by_address_space.end()) {
    EXPECT_EQ(proc_query_args.procs.size(),
              procs_mapped_by_address_space[test_address_space].size());
    for(const realm_processor_t proc : proc_query_args.procs) {
      EXPECT_TRUE(procs_mapped_by_address_space[test_address_space].find(Processor(
                      proc)) != procs_mapped_by_address_space[test_address_space].end());
    }
  } else {
    EXPECT_EQ(proc_query_args.procs.size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CProcessorQueryParamRestrictToAddressSpaceTestInstances,
    CProcessorQueryParamRestrictToAddressSpaceTest,
    ::testing::Values(
        // test case 1: two nodes, restrict address space to 0, expect to have 4
        // processors
        ProcessorQueryRestrictToAddressSpaceTestParam{
            std::vector<ProcessorQueryTestBaseParam>{
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::TOC_PROC, Processor::Kind::UTIL_PROC},
                 0},
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::OMP_PROC},
                 1}},
            0},
        // test case 2: two nodes, restrict address space to 2, expect to have no results
        ProcessorQueryRestrictToAddressSpaceTestParam{
            std::vector<ProcessorQueryTestBaseParam>{
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::TOC_PROC, Processor::Kind::UTIL_PROC},
                 0},
                {{Processor::Kind::LOC_PROC, Processor::Kind::LOC_PROC,
                  Processor::Kind::OMP_PROC},
                 1}},
            2}));
